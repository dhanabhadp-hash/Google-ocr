import express, { Request, Response } from 'express';
import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = Number(process.env.PORT) || 3000;

// Body parser with 50MB limit to handle large invoice images/PDFs
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Initialize Gemini Client
const geminiApiKey = process.env.GEMINI_API_KEY || '';
const ai = new GoogleGenAI({
  apiKey: geminiApiKey,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

// Helper for VAT analysis
function analyzeVat(subtotal: number, grandTotal: number, vatAmount: number) {
  if (subtotal <= 0 && grandTotal > 0) {
    subtotal = grandTotal;
  }
  const expectedVat = Math.round(subtotal * 0.07 * 100) / 100;
  const expectedGrandWithVat = Math.round((subtotal + expectedVat) * 100) / 100;

  // Check if grandTotal matches subtotal + 7%
  const diffFromAddedVat = Math.abs(grandTotal - expectedGrandWithVat);
  const diffFromSubtotal = Math.abs(grandTotal - subtotal);

  if (diffFromAddedVat < 1.0) {
    return {
      status: 'INCLUDED_7_PERCENT',
      rate: 7,
      note: 'บิลรวมภาษีมูลค่าเพิ่ม 7% ถูกต้อง (ราคาสินค้า + VAT 7%)',
      isCompliant: true,
    };
  }

  // Check if grandTotal already has 7% included inside subtotal
  if (vatAmount > 0 && Math.abs(vatAmount - Math.round((grandTotal * 7 / 107) * 100) / 100) < 1.0) {
    return {
      status: 'INCLUDED_7_PERCENT',
      rate: 7,
      note: 'ราคารายการรวมภาษีมูลค่าเพิ่ม 7% ไว้ในตัวแล้ว (VAT Included)',
      isCompliant: true,
    };
  }

  if (diffFromSubtotal < 1.0 && vatAmount === 0) {
    return {
      status: 'ZERO_OR_EXEMPT',
      rate: 0,
      note: 'รายการไม่คิดภาษีมูลค่าเพิ่ม หรือได้รับการยกเว้น VAT (Non-VAT / Exempt)',
      isCompliant: true,
    };
  }

  return {
    status: 'MISMATCH',
    rate: 7,
    note: `ยอดราคารวมอาจมีความคลาดเคลื่อน (ยอดที่คำนวณได้ไม่ตรงกับยอดบิล ต่างกันประมาณ ${Math.abs(grandTotal - expectedGrandWithVat).toFixed(2)} บาท)`,
    isCompliant: false,
  };
}

// Fallback & Retry helper to seamlessly handle 503 High Demand / UNAVAILABLE errors
async function callGeminiWithFallback(contents: any, config: any) {
  const candidateModels = [
    'gemini-3.8-flash',
    'gemini-3.1-flash-lite',
    'gemini-flash-latest',
    'gemini-2.5-flash',
  ];

  let lastError: any = null;

  for (const model of candidateModels) {
    for (let attempt = 1; attempt <= 2; attempt++) {
      try {
        console.log(`[Gemini OCR] Trying model ${model} (attempt ${attempt})...`);
        const response = await ai.models.generateContent({
          model,
          contents,
          config,
        });

        if (response && response.text) {
          console.log(`[Gemini OCR] Successfully generated with model ${model}`);
          return { response, modelUsed: model };
        }
      } catch (err: any) {
        lastError = err;
        const errMsg = err?.message || String(err);
        const errStatus = err?.status || err?.code || 0;
        console.warn(`[Gemini OCR] Model ${model} attempt ${attempt} returned error:`, errMsg);

        const isTransient =
          errStatus === 503 ||
          errStatus === 429 ||
          errMsg.includes('503') ||
          errMsg.includes('high demand') ||
          errMsg.includes('UNAVAILABLE') ||
          errMsg.includes('RESOURCE_EXHAUSTED') ||
          errMsg.includes('spikes in demand');

        if (isTransient) {
          if (attempt === 1) {
            // Wait 1.5 seconds and retry same model
            await new Promise((resolve) => setTimeout(resolve, 1500));
            continue;
          }
          // Switch to fallback model immediately
          break;
        } else {
          // If non-transient, also try next fallback model
          break;
        }
      }
    }
  }

  let finalMessage = lastError?.message || 'เกิดข้อผิดพลาดในการเชื่อมต่อกับโมเดล AI';
  try {
    const parsed = JSON.parse(finalMessage);
    if (parsed.error?.message) {
      finalMessage = parsed.error.message;
    }
  } catch {}

  if (
    finalMessage.includes('high demand') ||
    finalMessage.includes('503') ||
    finalMessage.includes('UNAVAILABLE') ||
    finalMessage.includes('spikes in demand')
  ) {
    finalMessage = 'ระบบ AI กำลังมีผู้ใช้งานหนาแน่นชั่วคราว (Spikes in demand) กรุณากดปุ่ม "ลองใหม่" อีกครั้ง';
  }

  throw new Error(finalMessage);
}

// POST /api/ocr - Process Invoice Image or PDF with Gemini 3.8 Flash
app.post('/api/ocr', async (req: Request, res: Response): Promise<void> => {
  try {
    const { fileBase64, mimeType, fileName } = req.body;

    if (!fileBase64 || !mimeType) {
      res.status(400).json({ error: 'Missing fileBase64 or mimeType' });
      return;
    }

    // Clean base64 string if it contains data URI prefix
    const cleanBase64 = fileBase64.replace(/^data:([a-zA-Z0-9_\-\.\/]+);base64,/, '');

    const systemPrompt = `คุณคือระบบ AI ผู้เชี่ยวชาญด้านการอ่านและสกัดข้อมูลเอกสารใบส่งของ/ใบกำกับภาษี/ใบเสร็จรับเงิน (Invoice / Tax Invoice / Delivery Note) สำหรับ "คลังยา โรงพยาบาลสบปราบ"
หน้าที่ของคุณคือ:
1. สกัดข้อมูลหัวบิล:
   - Invoice No (เลขที่บิล / ใบกำกับภาษี)
   - Company Name (ชื่อบริษัทผู้ผลิตหรือผู้จัดจำหน่ายยา เช่น บริษัท เบอร์ลินฟาร์มาซูติคอล, ซิลลิค ฟาร์มา, โพลีฟาร์ม, พาตาร์แล็บ, ไดอี, ชุมชนเภสัชกรรม ฯลฯ)
   - Tax ID (เลขประจำตัวผู้เสียภาษี 13 หลัก)
   - Invoice Date (วันที่ในบิล แปลงปี พ.ศ. เป็น ค.ศ. หากระบุ พ.ศ. เช่น 2567 -> 2024 หรือ คืนค่าในรูปแบบ YYYY-MM-DD หรือ DD/MM/YYYY)
   - Salesperson (ชื่อพนักงานขาย / ผู้แทนจำหน่าย หากมี)
   - Subtotal (มูลค่าสินค้าก่อนภาษี)
   - VAT Amount (ภาษีมูลค่าเพิ่ม 7% หากมี)
   - Grand Total (จำนวนเงินรวมทั้งสิ้น)

2. สกัดรายการยาในตาราง (Items) แต่ละรายการ:
   - GPU Code: รหัส GPU (รหัสกลุ่มยาสามัญ 5-8 หลัก หากมีในเอกสาร ถ้าไม่มีให้ปล่อยว่างหรือประมาณการตามมาตรฐาน)
   - TPU Code: รหัส TPU (รหัสยาการค้า 5-8 หลัก หากมีในบิล)
   - Product Name: ชื่อสินค้า / ชื่อยาการค้า (Trade Name) ตามที่ปรากฏในบิล พร้อมขนาด/ความแรง เช่น "Amoxil 500 mg cap", "Berlin Paracetamol 500mg"
   - Generic Name (ชื่อยาสามัญ): **วิเคราะห์และระบุชื่อยาสามัญสากล (INN / Generic Name)** จากชื่อยาการค้าที่สกัดได้ เช่น:
     - Amoxil -> Amoxicillin
     - Augmentin -> Amoxicillin + Clavulanic acid
     - Ponstan -> Mefenamic acid
     - Plavix -> Clopidogrel
     - Sara / Tylenol / GPO Paracet -> Paracetamol
     - Lipitor -> Atorvastatin
     - Norvasc -> Amlodipine
     - หากเป็นเวชภัณฑ์หรือน้ำเกลือ ให้ระบุชนิด เช่น 0.9% Sodium Chloride, Sterile Water for Irrigation
   - Quantity: จำนวน (ตัวเลข)
   - Unit Price: ราคาต่อหน่วย (ตัวเลข)
   - Total Price: ราคารวมของรายการ (ตัวเลข)

3. ตรวจสอบภาษีมูลค่าเพิ่ม VAT 7%:
   - วิเคราะห์ว่าบิลนี้มี VAT 7% หรือไม่ ยอด Grand Total รวมภาษีถูกต้องหรือไม่

ตอบกลับในรูปแบบ JSON เท่านั้น โดยไม่มีข้อความ Markdown อื่นๆ ครอบรูปแบบ schema ดังนี้:
{
  "invoiceNo": "string",
  "companyName": "string",
  "taxId": "string",
  "invoiceDate": "string",
  "salesperson": "string",
  "subtotal": 0,
  "vatAmount": 0,
  "grandTotal": 0,
  "items": [
    {
      "gpuCode": "string",
      "tpuCode": "string",
      "productName": "string",
      "genericName": "string",
      "quantity": 0,
      "unitPrice": 0,
      "totalPrice": 0
    }
  ],
  "rawNotes": "string"
}`;

    const promptText = `กรุณา OCR และสกัดข้อมูลจากเอกสารบิลยานี้ (ชื่อไฟล์: ${fileName || 'invoice'}) อย่างละเอียดและแม่นยำที่สุด สำหรับคลังยา รพ.สบปราบ`;

    const { response, modelUsed } = await callGeminiWithFallback(
      [
        {
          role: 'user',
          parts: [
            {
              inlineData: {
                mimeType: mimeType,
                data: cleanBase64,
              },
            },
            {
              text: `${systemPrompt}\n\n${promptText}`,
            },
          ],
        },
      ],
      {
        responseMimeType: 'application/json',
        temperature: 0.1,
      }
    );

    const text = response.text || '{}';
    let parsed: any;
    try {
      parsed = JSON.parse(text);
    } catch (e) {
      // Clean possible markdown code fences
      const clean = text.replace(/```json/g, '').replace(/```/g, '').trim();
      parsed = JSON.parse(clean);
    }

    // Calculate & verify VAT
    const subtotal = Number(parsed.subtotal || 0);
    const grandTotal = Number(parsed.grandTotal || 0);
    const vatAmount = Number(parsed.vatAmount || 0);
    const vatAnalysis = analyzeVat(subtotal, grandTotal, vatAmount);

    res.json({
      success: true,
      data: {
        invoiceNo: parsed.invoiceNo || 'INV-' + Date.now().toString().slice(-6),
        companyName: parsed.companyName || 'ไม่ระบุชื่อบริษัท',
        taxId: parsed.taxId || '',
        invoiceDate: parsed.invoiceDate || new Date().toISOString().split('T')[0],
        salesperson: parsed.salesperson || '-',
        subtotal: subtotal,
        vatAmount: vatAmount,
        grandTotal: grandTotal || subtotal,
        vatAnalysis: vatAnalysis,
        items: Array.isArray(parsed.items) ? parsed.items.map((it: any) => ({
          gpuCode: it.gpuCode || '',
          tpuCode: it.tpuCode || '',
          productName: it.productName || 'รายการยา',
          genericName: it.genericName || 'ไม่ระบุชื่อยาสามัญ',
          quantity: Number(it.quantity || 1),
          unitPrice: Number(it.unitPrice || 0),
          totalPrice: Number(it.totalPrice || 0),
        })) : [],
        rawNotes: parsed.rawNotes || '',
      },
    });
  } catch (error: any) {
    console.error('Error during OCR processing:', error);
    res.status(500).json({
      success: false,
      error: error?.message || 'เกิดข้อผิดพลาดในการประมวลผล OCR',
    });
  }
});

// POST /api/line/notify - Send Flex Message notification via LINE Messaging API
app.post('/api/line/notify', async (req: Request, res: Response): Promise<void> => {
  try {
    const { token, invoiceSummary } = req.body;
    const channelToken = token || process.env.LINE_CHANNEL_ACCESS_TOKEN;

    if (!channelToken) {
      res.status(400).json({
        success: false,
        error: 'กรุณาระบุ LINE Channel Access Token เพื่อส่งการแจ้งเตือน',
      });
      return;
    }

    const { invoiceNo, companyName, grandTotal, itemCount, vatStatus, date, driveUrl } = invoiceSummary;

    // Build standard LINE Flex Message
    const flexMessagePayload = {
      to: req.body.toUserId || undefined, // or broadcast endpoint if not specified
      messages: [
        {
          type: 'flex',
          altText: `บันทึกบิลคลังยาสำเร็จ: ${invoiceNo} (${companyName})`,
          contents: {
            type: 'bubble',
            size: 'mega',
            header: {
              type: 'box',
              layout: 'vertical',
              backgroundColor: '#059669',
              paddingAll: '16px',
              contents: [
                {
                  type: 'text',
                  text: '🏥 รพ.สบปราบ - คลังยา',
                  weight: 'bold',
                  color: '#ffffff',
                  size: 'sm',
                },
                {
                  type: 'text',
                  text: 'บันทึกบิล Invoice สำเร็จ',
                  weight: 'bold',
                  color: '#ffffff',
                  size: 'lg',
                  margin: 'xs',
                },
              ],
            },
            body: {
              type: 'box',
              layout: 'vertical',
              spacing: 'md',
              paddingAll: '16px',
              contents: [
                {
                  type: 'box',
                  layout: 'horizontal',
                  contents: [
                    { type: 'text', text: 'เลขที่บิล:', size: 'sm', color: '#64748b', flex: 3 },
                    { type: 'text', text: invoiceNo || '-', size: 'sm', weight: 'bold', color: '#1e293b', flex: 7 },
                  ],
                },
                {
                  type: 'box',
                  layout: 'horizontal',
                  contents: [
                    { type: 'text', text: 'ผู้จำหน่าย:', size: 'sm', color: '#64748b', flex: 3 },
                    { type: 'text', text: companyName || '-', size: 'sm', weight: 'bold', color: '#1e293b', flex: 7, wrap: true },
                  ],
                },
                {
                  type: 'box',
                  layout: 'horizontal',
                  contents: [
                    { type: 'text', text: 'วันที่ในบิล:', size: 'sm', color: '#64748b', flex: 3 },
                    { type: 'text', text: date || '-', size: 'sm', color: '#334155', flex: 7 },
                  ],
                },
                {
                  type: 'box',
                  layout: 'horizontal',
                  contents: [
                    { type: 'text', text: 'จำนวนรายการ:', size: 'sm', color: '#64748b', flex: 3 },
                    { type: 'text', text: `${itemCount || 1} รายการ`, size: 'sm', color: '#334155', flex: 7 },
                  ],
                },
                {
                  type: 'separator',
                  margin: 'md',
                },
                {
                  type: 'box',
                  layout: 'horizontal',
                  margin: 'md',
                  contents: [
                    { type: 'text', text: 'ยอดรวมสุทธิ:', size: 'md', weight: 'bold', color: '#0f172a', flex: 4 },
                    { type: 'text', text: `฿${Number(grandTotal || 0).toLocaleString('th-TH', { minimumFractionDigits: 2 })}`, size: 'lg', weight: 'bold', color: '#059669', align: 'end', flex: 6 },
                  ],
                },
                {
                  type: 'box',
                  layout: 'baseline',
                  contents: [
                    { type: 'text', text: 'สถานะ VAT:', size: 'xs', color: '#64748b', flex: 3 },
                    { type: 'text', text: vatStatus || 'ตรวจสอบแล้ว', size: 'xs', color: '#0284c7', flex: 7 },
                  ],
                },
              ],
            },
            footer: {
              type: 'box',
              layout: 'vertical',
              spacing: 'sm',
              paddingAll: '14px',
              contents: [
                {
                  type: 'button',
                  style: 'primary',
                  color: '#059669',
                  action: {
                    type: 'uri',
                    label: '📂 เปิดดูรูปบน Google Drive',
                    uri: driveUrl || 'https://drive.google.com/drive/folders/1_cwXmxuHj5enUq3VWNSWDlXTz_JpnwlY',
                  },
                },
                {
                  type: 'button',
                  style: 'link',
                  action: {
                    type: 'uri',
                    label: '📊 ดูชีทข้อมูล รพ.สบปราบ',
                    uri: 'https://docs.google.com/spreadsheets/d/1ogoM0vXPndiRcjNgbkN3Bsitd5JTcoT4rsj7hkerZqQ',
                  },
                },
              ],
            },
          },
        },
      ],
    };

    // If target userId provided, push message; otherwise broadcast
    const lineEndpoint = req.body.toUserId
      ? 'https://api.line.me/v2/bot/message/push'
      : 'https://api.line.me/v2/bot/message/broadcast';

    const lineResponse = await fetch(lineEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${channelToken}`,
      },
      body: JSON.stringify(flexMessagePayload),
    });

    const result = await lineResponse.json().catch(() => ({}));

    if (!lineResponse.ok) {
      res.status(lineResponse.status).json({
        success: false,
        error: result.message || 'เกิดข้อผิดพลาดในการส่ง LINE Message API',
        details: result,
      });
      return;
    }

    res.json({
      success: true,
      message: 'ส่งการแจ้งเตือน LINE สำเร็จ',
      result,
    });
  } catch (error: any) {
    console.error('Error sending LINE message:', error);
    res.status(500).json({
      success: false,
      error: error?.message || 'Failed to send LINE notification',
    });
  }
});

// POST /api/gas/proxy - Forward payload directly to Google Apps Script Web App
app.post('/api/gas/proxy', async (req: Request, res: Response): Promise<void> => {
  try {
    const { webAppUrl, payload } = req.body;
    const targetUrl = webAppUrl || process.env.GAS_WEBAPP_URL;

    if (!targetUrl) {
      res.status(400).json({
        success: false,
        error: 'กรุณาระบุ URL ของ Google Apps Script Web App ในการตั้งค่า',
      });
      return;
    }

    const gasResponse = await fetch(targetUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    const responseText = await gasResponse.text();
    let responseData: any;
    try {
      responseData = JSON.parse(responseText);
    } catch {
      responseData = { text: responseText };
    }

    res.json({
      success: gasResponse.ok,
      data: responseData,
    });
  } catch (error: any) {
    console.error('GAS proxy error:', error);
    res.status(500).json({
      success: false,
      error: error?.message || 'Error communicating with Google Apps Script',
    });
  }
});

// Health check endpoint
app.get('/api/health', (req: Request, res: Response) => {
  res.json({
    status: 'ok',
    appName: 'OCR INV. คลังยา รพ.สบปราบ',
    timestamp: new Date().toISOString(),
  });
});

// Mount Vite or static files
async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (req: Request, res: Response) => {
      res.sendFile(path.resolve(__dirname, 'dist/index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running at http://0.0.0.0:${PORT}`);
  });
}

startServer();
