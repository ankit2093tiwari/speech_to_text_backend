// Enhanced topic extraction module
const { GoogleGenerativeAI } = require("@google/generative-ai");

// Initialize Gemini AI (if API key is available)
let genAI = null;
try {
    if (process.env.GEMINI_API_KEY) {
        genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    }
} catch (error) {
    console.log("Gemini AI not available, using fallback methods");
}

// Common stop words to filter out
const STOP_WORDS = new Set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
    'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'yes'
]);

// Enhanced topic extraction using multiple methods
async function extractMainTopic(text) {
    console.log(`\n========== ENHANCED TOPIC EXTRACTION ==========`);
    console.log(`Input text: "${text.substring(0, 200)}..."`);

    // Method 1: Try Gemini AI first (most accurate)
    const geminiTopic = await extractTopicWithGemini(text);
    if (geminiTopic) {
        console.log(`✅ Gemini AI topic: "${geminiTopic}"`);
        return geminiTopic;
    }

    // Method 2: Frequency analysis with context awareness
    const frequencyTopic = extractTopicByFrequency(text);
    if (frequencyTopic) {
        console.log(`✅ Frequency analysis topic: "${frequencyTopic}"`);
        return frequencyTopic;
    }

    // Method 3: Pattern-based extraction
    const patternTopic = extractTopicByPatterns(text);
    if (patternTopic) {
        console.log(`✅ Pattern-based topic: "${patternTopic}"`);
        return patternTopic;
    }

    // Fallback: Use first few meaningful words
    const fallbackTopic = extractFallbackTopic(text);
    console.log(`⚠️ Fallback topic: "${fallbackTopic}"`);
    return fallbackTopic;
}

// Method 1: Use Gemini AI for intelligent topic extraction
async function extractTopicWithGemini(text) {
    if (!genAI) return null;

    try {
        const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

        const prompt = `
Analyze this speech transcript and identify the MAIN TOPIC the person is talking about.

Rules:
1. Look for what the speaker is primarily focused on
2. Pay attention to phrases like "I am talking about", "this is about", "the main thing is"
3. Consider frequency of mentions and descriptive content
4. Return only 1-3 words maximum
5. If multiple topics exist, choose the one with most emphasis/description

Transcript: "${text}"

Main topic (1-3 words only):`;

        const result = await model.generateContent(prompt);
        const response = result.response;
        const topic = response.text().trim().toLowerCase();

        // Validate the response (should be 1-3 words)
        const words = topic.split(/\s+/).filter(word => word.length > 0);
        if (words.length <= 3 && words.length > 0) {
            return words.join(' ');
        }

        return null;
    } catch (error) {
        console.error("Gemini AI error:", error);
        return null;
    }
}

// Method 2: Enhanced frequency analysis with context
function extractTopicByFrequency(text) {
    const words = text.toLowerCase()
        .replace(/[^\w\s]/g, ' ')
        .split(/\s+/)
        .filter(word => word.length > 2 && !STOP_WORDS.has(word));

    // Count word frequencies
    const wordCount = {};
    words.forEach(word => {
        wordCount[word] = (wordCount[word] || 0) + 1;
    });

    // Look for explicit topic indicators with different weights
    const topicIndicators = [
        { pattern: /(?:i am (?:objectively )?talking about)\s+(?:the\s+)?(\w+)/gi, weight: 10 },
        { pattern: /(?:this is about|it is about)\s+(?:the\s+)?(\w+)/gi, weight: 8 },
        { pattern: /(?:i (?:like|love|prefer))\s+(?:the\s+)?(\w+)/gi, weight: 6 },
        { pattern: /(?:talking about|discussing|mentioning)\s+(?:the\s+)?(\w+)/gi, weight: 5 },
        { pattern: /(?:so i like)\s+(?:the\s+)?(\w+)/gi, weight: 7 }
    ];

    for (const indicator of topicIndicators) {
        const matches = [...text.matchAll(indicator.pattern)];
        for (const match of matches) {
            const word = match[1].toLowerCase();
            if (word.length > 2 && !STOP_WORDS.has(word)) {
                // Boost score based on indicator weight
                wordCount[word] = (wordCount[word] || 0) + indicator.weight;
            }
        }
    }

    // Find words that appear multiple times
    const frequentWords = Object.entries(wordCount)
        .filter(([word, count]) => count >= 2)
        .sort((a, b) => b[1] - a[1]);

    if (frequentWords.length > 0) {
        return frequentWords[0][0];
    }

    return null;
}

// Method 3: Pattern-based extraction for common speech patterns
function extractTopicByPatterns(text) {
    const patterns = [
        // Direct topic statements (highest priority)
        /(?:i am (?:objectively )?talking about (?:the )?)([\w]+)/gi,
        /(?:this is about (?:the )?)([\w]+)/gi,
        /(?:the main (?:thing|topic|subject) is (?:the )?)([\w]+)/gi,
        /(?:let me tell you about (?:the )?)([\w]+)/gi,

        // Preference statements
        /(?:i (?:like|love|prefer) (?:the )?)([\w]+)/gi,
        /(?:my favorite (?:is )?(?:the )?)([\w]+)/gi,

        // Descriptive patterns
        /(?:the )([\w]+)(?: is| are| was| were) (?:very|really|so|quite)/gi,
        /(?:this )([\w]+)(?: is| was)/gi,

        // Emphasis patterns
        /(?:yes\.? (?:this|that|it) is (?:a )?(?:very )?(?:the )?)([\w]+)/gi,
        /(?:so i like (?:the )?)([\w]+)/gi
    ];

    // Track pattern matches with scores
    const patternMatches = {};

    for (let i = 0; i < patterns.length; i++) {
        const pattern = patterns[i];
        const matches = [...text.matchAll(pattern)];

        for (const match of matches) {
            const topic = match[1].trim().toLowerCase();

            if (topic.length > 2 && !STOP_WORDS.has(topic)) {
                // Higher score for earlier patterns (more explicit statements)
                const score = patterns.length - i;
                patternMatches[topic] = (patternMatches[topic] || 0) + score;
            }
        }
    }

    // Return the topic with highest pattern score
    if (Object.keys(patternMatches).length > 0) {
        const bestTopic = Object.entries(patternMatches)
            .sort((a, b) => b[1] - a[1])[0][0];
        return bestTopic;
    }

    return null;
}

// Fallback method: Extract meaningful words
function extractFallbackTopic(text) {
    const words = text.toLowerCase()
        .replace(/[^\w\s]/g, ' ')
        .split(/\s+/)
        .filter(word => word.length > 3 && !STOP_WORDS.has(word));

    // Return first meaningful word or combination
    if (words.length > 0) {
        return words[0];
    }

    return "unknown topic";
}

module.exports = {
    extractMainTopic
};