import { NextResponse } from 'next/server';
import axios from 'axios';

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: CORS_HEADERS,
  });
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { 
      action, 
      question_text, 
      option_A, 
      option_B, 
      option_C, 
      option_D, 
      solution,
      correct_option,
      model = 'llama-3.3-70b-versatile', 
      temperature = 0.3, 
      max_tokens = 500 
    } = body || {};

    // ==================== GENERATE SOLUTION ====================
    if (action === 'generate_solution') {
      if (!question_text || !option_A || !option_B || !option_C || !option_D || !solution) {
        return new NextResponse(
          JSON.stringify({ error: 'question_text, all options, and solution are required' }), 
          {
            status: 400,
            headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
          }
        );
      }

      const prompt = `You are an expert JEE exam tutor. Given the question, options, and the solution logic, create a clean, well-structured, and easy-to-understand solution.

**Requirements:**
1. Break down the solution into clear, numbered steps
2. Use proper spacing and structure (NOT a paragraph)
3. Include LaTeX formatting for mathematical expressions using $ for inline math and $$ for block equations
4. Make each step easy to follow
5. Add brief explanations where needed
6. Use bullet points or numbered lists for clarity
7. Highlight key formulas or concepts

**Question:** ${question_text}

**Options:**
A) ${option_A}
B) ${option_B}
C) ${option_C}
D) ${option_D}

**Correct Answer:** ${correct_option || 'To be determined'}

**Solution Logic:** ${solution}

**Generate a clean, structured solution following the requirements above:**`;

      const groqRes = await axios.post(
        'https://api.groq.com/openai/v1/chat/completions',
        {
          model,
          messages: [{ role: 'user', content: prompt }],
          temperature: 0.5,
          max_tokens: 2000,
        },
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
          },
          timeout: 120000,
        }
      );

      const generatedSolution = groqRes.data.choices?.[0]?.message?.content || solution;

      return new NextResponse(
        JSON.stringify({ 
          solution: generatedSolution,
          full_response: groqRes.data 
        }), 
        {
          status: 200,
          headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
        }
      );
    }

    // ==================== DETERMINE ANSWER ====================
    else if (action === 'determine_answer') {
      if (!question_text || !option_A || !option_B || !option_C || !option_D || !solution) {
        return new NextResponse(
          JSON.stringify({ error: 'question_text, all options (A, B, C, D), and solution are required' }), 
          {
            status: 400,
            headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
          }
        );
      }

      const prompt = `You are a JEE exam expert. Given the following question and solution, determine the correct answer option.

Question: ${question_text}

Option A: ${option_A}
Option B: ${option_B}
Option C: ${option_C}
Option D: ${option_D}

Solution: ${solution}

Based on the solution provided, which option (A, B, C, or D) is the correct answer? 

IMPORTANT: Respond with ONLY a single letter: A, B, C, or D. Do not include any explanation, punctuation, or additional text.`;

      const groqRes = await axios.post(
        'https://api.groq.com/openai/v1/chat/completions',
        {
          model,
          messages: [{ role: 'user', content: prompt }],
          temperature: 0.1,
          max_tokens: 10,
        },
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
          },
          timeout: 120000,
        }
      );

      const aiResponse = groqRes.data.choices?.[0]?.message?.content || '';
      const correctAnswer = aiResponse.trim().toUpperCase().match(/[ABCD]/)?.[0] || null;

      if (!correctAnswer) {
        return new NextResponse(
          JSON.stringify({ error: 'Could not determine correct answer from AI response', aiResponse }), 
          {
            status: 500,
            headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
          }
        );
      }

      return new NextResponse(
        JSON.stringify({ 
          correct_answer: correctAnswer,
          raw_response: aiResponse 
        }), 
        {
          status: 200,
          headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
        }
      );
    }

    // ==================== EXPLAIN LIKE 5 YEAR OLD ====================
    else if (action === 'explain_5yr') {
      if (!question_text || !solution) {
        return new NextResponse(
          JSON.stringify({ error: 'question_text and solution are required for explain_5yr action' }), 
          {
            status: 400,
            headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
          }
        );
      }

      const prompt = `Explain the following JEE question solution in very simple terms that even a 5-year-old could understand. Use analogies, simple language, and avoid technical jargon.

Question: ${question_text}
Solution: ${solution}

Explain this solution in simple, friendly language:`;

      const groqRes = await axios.post(
        'https://api.groq.com/openai/v1/chat/completions',
        {
          model,
          messages: [{ role: 'user', content: prompt }],
          temperature: 0.7,
          max_tokens: 800,
        },
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
          },
          timeout: 120000,
        }
      );

      const explanation = groqRes.data.choices?.[0]?.message?.content || 'Could not generate explanation.';

      return new NextResponse(
        JSON.stringify({ 
          explanation,
          full_response: groqRes.data 
        }), 
        {
          status: 200,
          headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
        }
      );
    }

    // ==================== BETTER UNDERSTANDING ====================
    else if (action === 'better_understanding') {
      if (!question_text || !solution) {
        return new NextResponse(
          JSON.stringify({ error: 'question_text and solution are required for better_understanding action' }), 
          {
            status: 400,
            headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
          }
        );
      }

      const prompt = `Provide a simpler, more intuitive explanation of this JEE question solution. Focus on the core concept and make it easier to understand.

Question: ${question_text}
Solution: ${solution}

Provide a clearer, more intuitive explanation:`;

      const groqRes = await axios.post(
        'https://api.groq.com/openai/v1/chat/completions',
        {
          model,
          messages: [{ role: 'user', content: prompt }],
          temperature: 0.7,
          max_tokens: 800,
        },
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
          },
          timeout: 120000,
        }
      );

      const explanation = groqRes.data.choices?.[0]?.message?.content || 'Could not generate explanation.';

      return new NextResponse(
        JSON.stringify({ 
          explanation,
          full_response: groqRes.data 
        }), 
        {
          status: 200,
          headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
        }
      );
    }

    // ==================== DIG DEEPER ====================
    else if (action === 'dig_deeper') {
      if (!question_text || !solution) {
        return new NextResponse(
          JSON.stringify({ error: 'question_text and solution are required for dig_deeper action' }), 
          {
            status: 400,
            headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
          }
        );
      }

      const prompt = `Based on this JEE question and solution, create a simpler conceptual MCQ to test understanding of the underlying concept. Make it easier than the original question but focus on the same core concept.

Question: ${question_text}
Solution: ${solution}

Create a new MCQ and respond in this EXACT JSON format (no additional text):
{
  "question": "your question here",
  "options": ["option 1", "option 2", "option 3", "option 4"],
  "correctAnswer": "A",
  "explanation": "brief explanation of why this is the correct answer"
}`;

      const groqRes = await axios.post(
        'https://api.groq.com/openai/v1/chat/completions',
        {
          model,
          messages: [{ role: 'user', content: prompt }],
          temperature: 0.7,
          max_tokens: 1000,
        },
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
          },
          timeout: 120000,
        }
      );

      const aiResponse = groqRes.data.choices?.[0]?.message?.content || '';
      
      const jsonMatch = aiResponse.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        return new NextResponse(
          JSON.stringify({ error: 'Could not parse MCQ JSON from AI response', aiResponse }), 
          {
            status: 500,
            headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
          }
        );
      }

      const mcqData = JSON.parse(jsonMatch[0]);

      return new NextResponse(
        JSON.stringify({ 
          mcq: mcqData,
          raw_response: aiResponse 
        }), 
        {
          status: 200,
          headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
        }
      );
    }

    // ==================== DEFAULT (Backward Compatibility) ====================
    else {
      const { message } = body;
      
      if (!message) {
        return new NextResponse(
          JSON.stringify({ error: 'message is required for default action' }), 
          {
            status: 400,
            headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
          }
        );
      }

      const groqRes = await axios.post(
        'https://api.groq.com/openai/v1/chat/completions',
        {
          model,
          messages: [{ role: 'user', content: message }],
          temperature,
          max_tokens,
        },
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
          },
          timeout: 120000,
        }
      );

      return new NextResponse(JSON.stringify(groqRes.data), {
        status: 200,
        headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
      });
    }
  } catch (err: any) {
    console.error('solution route error:', err?.response?.data || err?.message || err);
    const status = err?.response?.status || 500;
    const details = err?.response?.data || err?.message || 'Unknown error';
    return new NextResponse(
      JSON.stringify({ error: 'Failed to process request', details }), 
      {
        status,
        headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
      }
    );
  }
}