from __future__ import annotations

import json
from pathlib import Path


SOURCE = "synthetic_multilingual_v1"
INSTRUCTION_PATH = Path("data/raw/indoone_instructions.jsonl")
MULTILINGUAL_PATH = Path("data/raw/indoone_multilingual_examples.jsonl")
CORPUS_PATH = Path("data/raw/indoone_corpus.txt")

# This is deterministic augmentation, not a replacement for curated human data.
# It is intentionally balanced across the initial Indian-language target set so
# training infrastructure can exercise multilingual paths while the curated
# corpus is expanded over time.
LANGUAGES: dict[str, dict[str, str]] = {
    "kn": {
        "add": "{a} + {b} ಎಷ್ಟು?", "add_r": "{a} + {b} = {r}.",
        "sub": "{a} ಯಿಂದ {b} ಕಡಿತ ಮಾಡಿದರೆ ಎಷ್ಟು?", "sub_r": "{a} - {b} = {r}.",
        "mul": "{a} ಅನ್ನು {b}ರಿಂದ ಗುಣಿಸಿದರೆ ಎಷ್ಟು?", "mul_r": "{a} × {b} = {r}.",
        "pct": "{a} ರ ಶೇಕಡಾ {b} ಎಷ್ಟು?", "pct_r": "{a} ರ {b}% = {r}.",
        "even": "{a} ಸಮ ಸಂಖ್ಯೆಯೇ?", "even_r": "ಹೌದು, {a} ಒಂದು ಸಮ ಸಂಖ್ಯೆ.",
        "odd": "{a} ಬೆಸ ಸಂಖ್ಯೆಯೇ?", "odd_r": "ಹೌದು, {a} ಒಂದು ಬೆಸ ಸಂಖ್ಯೆ.",
        "compare": "{a} ಮತ್ತು {b}ರಲ್ಲಿ ದೊಡ್ಡದು ಯಾವುದು?", "compare_r": "ದೊಡ್ಡ ಸಂಖ್ಯೆ {r}.",
        "clarify": "ಅಸ್ಪಷ್ಟ ಪ್ರಶ್ನೆಗೆ ಏನು ಮಾಡಬೇಕು?", "clarify_r": "ಮುಖ್ಯವಾದ ವಿವರವನ್ನು ಕೇಳಿ, ಅಸುರಕ್ಷಿತ ಊಹೆ ಮಾಡಬೇಡಿ.",
        "learn": "ಯಂತ್ರ ಕಲಿಕೆ ಎಂದರೇನು?", "learn_r": "ಯಂತ್ರ ಕಲಿಕೆ ಎಂದರೆ ದತ್ತಾಂಶದಿಂದ ಮಾದರಿಗಳನ್ನು ಕಲಿತು ಹೊಸ ಒಳಹರಿವುಗಳ ಮೇಲೆ ನಿರ್ಧಾರ ಅಥವಾ ಊಹೆ ಮಾಡುವ ವಿಧಾನ.",
        "verify": "ತಿಳಿಯದ ಉತ್ತರ ಬಂದಾಗ ಏನು ಮಾಡಬೇಕು?", "verify_r": "ತಿಳಿಯುವುದಿಲ್ಲ ಎಂದು ಸ್ಪಷ್ಟವಾಗಿ ಹೇಳಿ ಮತ್ತು ಪರಿಶೀಲಿಸದೆ ಮಾಹಿತಿ ಕಲ್ಪಿಸಬೇಡಿ.",
        "translate": "ಅನುವಾದದಲ್ಲಿ ಸಂಖ್ಯೆಗಳನ್ನು ಹೇಗೆ ಉಳಿಸಬೇಕು?", "translate_r": "ಸಂಖ್ಯೆಗಳು, ಹೆಸರುಗಳು ಮತ್ತು ದಿನಾಂಕಗಳನ್ನು ನಿಖರವಾಗಿ ಉಳಿಸಬೇಕು.",
        "safety": "ಅಸುರಕ್ಷಿತ ವಿಧಾನ ಕೇಳಿದಾಗ ಏನು ಮಾಡಬೇಕು?", "safety_r": "ಅಸುರಕ್ಷಿತ ವಿಧಾನವನ್ನು ನೀಡದೆ ಸುರಕ್ಷಿತ ಪರ್ಯಾಯವನ್ನು ಸೂಚಿಸಬೇಕು.",
    },
    "hi": {
        "add": "{a} + {b} कितना है?", "add_r": "{a} + {b} = {r}।",
        "sub": "{a} में से {b} घटाने पर कितना होगा?", "sub_r": "{a} - {b} = {r}।",
        "mul": "{a} को {b} से गुणा करने पर कितना होगा?", "mul_r": "{a} × {b} = {r}।",
        "pct": "{a} का {b} प्रतिशत कितना है?", "pct_r": "{a} का {b}% = {r}।",
        "even": "क्या {a} सम संख्या है?", "even_r": "हाँ, {a} एक सम संख्या है।",
        "odd": "क्या {a} विषम संख्या है?", "odd_r": "हाँ, {a} एक विषम संख्या है।",
        "compare": "{a} और {b} में बड़ी संख्या कौन सी है?", "compare_r": "बड़ी संख्या {r} है।",
        "clarify": "अस्पष्ट प्रश्न पर क्या करना चाहिए?", "clarify_r": "जो महत्वपूर्ण जानकारी नहीं है उसे पूछें और जोखिम भरा अनुमान न लगाएँ।",
        "learn": "मशीन लर्निंग क्या है?", "learn_r": "मशीन लर्निंग वह तरीका है जिसमें प्रणाली डेटा से पैटर्न सीखकर नए इनपुट पर अनुमान या निर्णय करती है।",
        "verify": "उत्तर न पता हो तो क्या करना चाहिए?", "verify_r": "स्पष्ट रूप से कहें कि उत्तर ज्ञात नहीं है और बिना जाँच के जानकारी न गढ़ें।",
        "translate": "अनुवाद में संख्याएँ कैसे रखनी चाहिए?", "translate_r": "संख्याओं, नामों और तारीखों को सटीक रूप से बनाए रखना चाहिए।",
        "safety": "असुरक्षित तरीका माँगा जाए तो क्या करना चाहिए?", "safety_r": "असुरक्षित तरीका न दें और जहाँ संभव हो सुरक्षित विकल्प दें।",
    },
    "te": {
        "add": "{a} + {b} ఎంత?", "add_r": "{a} + {b} = {r}.",
        "sub": "{a} నుండి {b} తీసేస్తే ఎంత?", "sub_r": "{a} - {b} = {r}.",
        "mul": "{a} ను {b} తో గుణిస్తే ఎంత?", "mul_r": "{a} × {b} = {r}.",
        "pct": "{a} యొక్క {b} శాతం ఎంత?", "pct_r": "{a} యొక్క {b}% = {r}.",
        "even": "{a} సరి సంఖ్యా?", "even_r": "అవును, {a} సరి సంఖ్య.",
        "odd": "{a} బేసి సంఖ్యా?", "odd_r": "అవును, {a} బేసి సంఖ్య.",
        "compare": "{a}, {b} లో పెద్దది ఏది?", "compare_r": "పెద్ద సంఖ్య {r}.",
        "clarify": "అస్పష్టమైన ప్రశ్న వస్తే ఏమి చేయాలి?", "clarify_r": "ప్రధానంగా అవసరమైన వివరాన్ని అడిగి, ప్రమాదకరమైన ఊహ చేయకూడదు.",
        "learn": "మెషిన్ లెర్నింగ్ అంటే ఏమిటి?", "learn_r": "మెషిన్ లెర్నింగ్ అనేది డేటా నుంచి నమూనాలను నేర్చుకుని కొత్త ఇన్‌పుట్‌లపై అంచనా లేదా నిర్ణయం చేసే విధానం.",
        "verify": "సమాధానం తెలియకపోతే ఏమి చేయాలి?", "verify_r": "తెలియదని స్పష్టంగా చెప్పి, ధృవీకరించకుండా సమాచారం కల్పించకూడదు.",
        "translate": "అనువాదంలో సంఖ్యలను ఎలా ఉంచాలి?", "translate_r": "సంఖ్యలు, పేర్లు, తేదీలను ఖచ్చితంగా ఉంచాలి.",
        "safety": "అసురక్షిత పద్ధతి అడిగితే ఏమి చేయాలి?", "safety_r": "అసురక్షిత పద్ధతిని ఇవ్వకుండా సురక్షితమైన ప్రత్యామ్నాయాన్ని సూచించాలి.",
    },
    "ta": {
        "add": "{a} + {b} எவ்வளவு?", "add_r": "{a} + {b} = {r}.",
        "sub": "{a} இலிருந்து {b} கழித்தால் எவ்வளவு?", "sub_r": "{a} - {b} = {r}.",
        "mul": "{a} ஐ {b} ஆல் பெருக்கினால் எவ்வளவு?", "mul_r": "{a} × {b} = {r}.",
        "pct": "{a} இன் {b} சதவீதம் எவ்வளவு?", "pct_r": "{a} இன் {b}% = {r}.",
        "even": "{a} இரட்டை எண்ணா?", "even_r": "ஆம், {a} ஒரு இரட்டை எண்.",
        "odd": "{a} ஒற்றை எண்ணா?", "odd_r": "ஆம், {a} ஒரு ஒற்றை எண்.",
        "compare": "{a}, {b} இல் பெரியது எது?", "compare_r": "பெரிய எண் {r}.",
        "clarify": "தெளிவில்லாத கேள்விக்கு என்ன செய்ய வேண்டும்?", "clarify_r": "முக்கியமான விடுபட்ட விவரத்தை கேட்டு, ஆபத்தான ஊகத்தை தவிர்க்க வேண்டும்.",
        "learn": "இயந்திரக் கற்றல் என்றால் என்ன?", "learn_r": "இயந்திரக் கற்றல் என்பது தரவிலிருந்து வடிவங்களை கற்று புதிய உள்ளீடுகளில் கணிப்பு அல்லது முடிவு செய்வது.",
        "verify": "பதில் தெரியாவிட்டால் என்ன செய்ய வேண்டும்?", "verify_r": "தெரியாது என்று தெளிவாகச் சொல்லி, சரிபார்க்காமல் தகவலை உருவாக்கக்கூடாது.",
        "translate": "மொழிபெயர்ப்பில் எண்களை எப்படி பாதுகாக்க வேண்டும்?", "translate_r": "எண்கள், பெயர்கள், தேதிகளை துல்லியமாக பாதுகாக்க வேண்டும்.",
        "safety": "பாதுகாப்பற்ற முறையை கேட்டால் என்ன செய்ய வேண்டும்?", "safety_r": "பாதுகாப்பற்ற முறையை வழங்காமல் பாதுகாப்பான மாற்றை முன்மொழிய வேண்டும்.",
    },
    "ml": {
        "add": "{a} + {b} എത്ര?", "add_r": "{a} + {b} = {r}.",
        "sub": "{a}ൽ നിന്ന് {b} കുറച്ചാൽ എത്ര?", "sub_r": "{a} - {b} = {r}.",
        "mul": "{a} നെ {b} കൊണ്ട് ഗുണിച്ചാൽ എത്ര?", "mul_r": "{a} × {b} = {r}.",
        "pct": "{a} ന്റെ {b} ശതമാനം എത്ര?", "pct_r": "{a} ന്റെ {b}% = {r}.",
        "even": "{a} സമ സംഖ്യയാണോ?", "even_r": "അതെ, {a} ഒരു സമ സംഖ്യയാണ്.",
        "odd": "{a} ഒറ്റ സംഖ്യയാണോ?", "odd_r": "അതെ, {a} ഒരു ഒറ്റ സംഖ്യയാണ്.",
        "compare": "{a}നും {b}നും ഇടയിൽ വലിയത് ഏത്?", "compare_r": "വലിയ സംഖ്യ {r} ആണ്.",
        "clarify": "അവ്യക്തമായ ചോദ്യത്തിൽ എന്ത് ചെയ്യണം?", "clarify_r": "പ്രധാനമായ ആവശ്യമായ വിശദാംശം ചോദിക്കുകയും അപകടകരമായ അനുമാനം ഒഴിവാക്കുകയും വേണം.",
        "learn": "മെഷീൻ ലേണിംഗ് എന്താണ്?", "learn_r": "ഡാറ്റയിൽ നിന്ന് മാതൃകകൾ പഠിച്ച് പുതിയ ഇൻപുട്ടുകളിൽ പ്രവചനം അല്ലെങ്കിൽ തീരുമാനം ചെയ്യുന്നതാണ് മെഷീൻ ലേണിംഗ്.",
        "verify": "ഉത്തരം അറിയില്ലെങ്കിൽ എന്ത് ചെയ്യണം?", "verify_r": "അറിയില്ലെന്ന് വ്യക്തമായി പറയുകയും പരിശോധിക്കാതെ വിവരം സൃഷ്ടിക്കാതിരിക്കുകയും വേണം.",
        "translate": "പരിഭാഷയിൽ സംഖ്യകൾ എങ്ങനെ നിലനിർത്തണം?", "translate_r": "സംഖ്യകളും പേരുകളും തീയതികളും കൃത്യമായി നിലനിർത്തണം.",
        "safety": "അസുരക്ഷിതമായ രീതി ആവശ്യപ്പെട്ടാൽ എന്ത് ചെയ്യണം?", "safety_r": "അസുരക്ഷിതമായ രീതി നൽകാതെ സുരക്ഷിതമായൊരു വഴി നിർദ്ദേശിക്കണം.",
    },
    "mr": {
        "add": "{a} + {b} किती?", "add_r": "{a} + {b} = {r}.",
        "sub": "{a} मधून {b} वजा केल्यावर किती?", "sub_r": "{a} - {b} = {r}.",
        "mul": "{a} ला {b} ने गुणल्यावर किती?", "mul_r": "{a} × {b} = {r}.",
        "pct": "{a} चा {b} टक्के किती?", "pct_r": "{a} चा {b}% = {r}.",
        "even": "{a} सम संख्या आहे का?", "even_r": "होय, {a} सम संख्या आहे.",
        "odd": "{a} विषम संख्या आहे का?", "odd_r": "होय, {a} विषम संख्या आहे.",
        "compare": "{a} आणि {b} पैकी मोठी संख्या कोणती?", "compare_r": "मोठी संख्या {r} आहे.",
        "clarify": "अस्पष्ट प्रश्न आल्यास काय करावे?", "clarify_r": "महत्त्वाचा गहाळ तपशील विचारा आणि धोकादायक अंदाज टाळा.",
        "learn": "मशीन लर्निंग म्हणजे काय?", "learn_r": "मशीन लर्निंग म्हणजे डेटामधून नमुने शिकून नवीन इनपुटवर अंदाज किंवा निर्णय घेण्याची पद्धत.",
        "verify": "उत्तर माहीत नसल्यास काय करावे?", "verify_r": "उत्तर माहीत नाही असे स्पष्ट सांगा आणि तपासणीशिवाय माहिती बनवू नका.",
        "translate": "भाषांतरात संख्या कशा जपाव्यात?", "translate_r": "संख्या, नावे आणि तारखा अचूकपणे जपाव्यात.",
        "safety": "असुरक्षित पद्धत विचारली तर काय करावे?", "safety_r": "असुरक्षित पद्धत देऊ नका आणि सुरक्षित पर्याय सुचवा.",
    },
    "bn": {
        "add": "{a} + {b} কত?", "add_r": "{a} + {b} = {r}।",
        "sub": "{a} থেকে {b} বাদ দিলে কত?", "sub_r": "{a} - {b} = {r}।",
        "mul": "{a} কে {b} দিয়ে গুণ করলে কত?", "mul_r": "{a} × {b} = {r}।",
        "pct": "{a} এর {b} শতাংশ কত?", "pct_r": "{a} এর {b}% = {r}।",
        "even": "{a} জোড় সংখ্যা কি?", "even_r": "হ্যাঁ, {a} একটি জোড় সংখ্যা।",
        "odd": "{a} বিজোড় সংখ্যা কি?", "odd_r": "হ্যাঁ, {a} একটি বিজোড় সংখ্যা।",
        "compare": "{a} এবং {b} এর মধ্যে বড় কোনটি?", "compare_r": "বড় সংখ্যা হল {r}।",
        "clarify": "অস্পষ্ট প্রশ্ন হলে কী করা উচিত?", "clarify_r": "প্রয়োজনীয় গুরুত্বপূর্ণ তথ্য জিজ্ঞাসা করুন এবং ঝুঁকিপূর্ণ অনুমান করবেন না।",
        "learn": "মেশিন লার্নিং কী?", "learn_r": "মেশিন লার্নিং হলো ডেটা থেকে ধরণ শিখে নতুন ইনপুটে অনুমান বা সিদ্ধান্ত করার পদ্ধতি।",
        "verify": "উত্তর জানা না থাকলে কী করা উচিত?", "verify_r": "জানা নেই বলে স্পষ্টভাবে জানান এবং যাচাই ছাড়া তথ্য বানাবেন না।",
        "translate": "অনুবাদে সংখ্যা কীভাবে রাখা উচিত?", "translate_r": "সংখ্যা, নাম এবং তারিখ সঠিকভাবে রাখতে হবে।",
        "safety": "অনিরাপদ পদ্ধতি চাইলে কী করা উচিত?", "safety_r": "অনিরাপদ পদ্ধতি না দিয়ে নিরাপদ বিকল্প প্রস্তাব করা উচিত।",
    },
    "gu": {
        "add": "{a} + {b} કેટલું?", "add_r": "{a} + {b} = {r}.",
        "sub": "{a} માંથી {b} બાદ કરતાં કેટલું?", "sub_r": "{a} - {b} = {r}.",
        "mul": "{a} ને {b} થી ગુણીએ તો કેટલું?", "mul_r": "{a} × {b} = {r}.",
        "pct": "{a} નું {b} ટકા કેટલું?", "pct_r": "{a} નું {b}% = {r}.",
        "even": "શું {a} સમ સંખ્યા છે?", "even_r": "હા, {a} સમ સંખ્યા છે.",
        "odd": "શું {a} વિષમ સંખ્યા છે?", "odd_r": "હા, {a} વિષમ સંખ્યા છે.",
        "compare": "{a} અને {b}માંથી મોટી સંખ્યા કઈ?", "compare_r": "મોટી સંખ્યા {r} છે.",
        "clarify": "અસ્પષ્ટ પ્રશ્ન હોય તો શું કરવું?", "clarify_r": "જરૂરી માહિતી પૂછો અને જોખમી અનુમાન ન કરો.",
        "learn": "મશીન લર્નિંગ શું છે?", "learn_r": "મશીન લર્નિંગ એ ડેટામાંથી નમૂનાઓ શીખીને નવા ઇનપુટ પર આગાહી અથવા નિર્ણય કરવાની પદ્ધતિ છે.",
        "verify": "જવાબ ખબર ન હોય તો શું કરવું?", "verify_r": "જવાબ ખબર નથી એવું સ્પષ્ટ કહો અને ચકાસ્યા વિના માહિતી ન બનાવો.",
        "translate": "અનુવાદમાં સંખ્યાઓ કેવી રીતે રાખવી?", "translate_r": "સંખ્યાઓ, નામો અને તારીખો ચોક્કસ રીતે જાળવવી જોઈએ.",
        "safety": "અસુરક્ષિત રીત પૂછવામાં આવે તો શું કરવું?", "safety_r": "અસુરક્ષિત રીત ન આપો અને સુરક્ષિત વિકલ્પ સૂચવો.",
    },
    "pa": {
        "add": "{a} + {b} ਕਿੰਨਾ ਹੈ?", "add_r": "{a} + {b} = {r}।",
        "sub": "{a} ਵਿੱਚੋਂ {b} ਘਟਾਉਣ ਤੇ ਕਿੰਨਾ?", "sub_r": "{a} - {b} = {r}।",
        "mul": "{a} ਨੂੰ {b} ਨਾਲ ਗੁਣਾ ਕਰਨ ਤੇ ਕਿੰਨਾ?", "mul_r": "{a} × {b} = {r}।",
        "pct": "{a} ਦਾ {b} ਪ੍ਰਤੀਸ਼ਤ ਕਿੰਨਾ ਹੈ?", "pct_r": "{a} ਦਾ {b}% = {r}।",
        "even": "ਕੀ {a} ਸਮ ਸੰਖਿਆ ਹੈ?", "even_r": "ਹਾਂ, {a} ਸਮ ਸੰਖਿਆ ਹੈ।",
        "odd": "ਕੀ {a} ਵਿਸਮ ਸੰਖਿਆ ਹੈ?", "odd_r": "ਹਾਂ, {a} ਵਿਸਮ ਸੰਖਿਆ ਹੈ।",
        "compare": "{a} ਅਤੇ {b} ਵਿੱਚ ਵੱਡੀ ਸੰਖਿਆ ਕਿਹੜੀ ਹੈ?", "compare_r": "ਵੱਡੀ ਸੰਖਿਆ {r} ਹੈ।",
        "clarify": "ਅਸਪਸ਼ਟ ਸਵਾਲ ਹੋਵੇ ਤਾਂ ਕੀ ਕਰਨਾ ਚਾਹੀਦਾ ਹੈ?", "clarify_r": "ਲੋੜੀਂਦਾ ਵੇਰਵਾ ਪੁੱਛੋ ਅਤੇ ਖਤਰਨਾਕ ਅਨੁਮਾਨ ਨਾ ਲਗਾਓ।",
        "learn": "ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਕੀ ਹੈ?", "learn_r": "ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਉਹ ਤਰੀਕਾ ਹੈ ਜਿਸ ਵਿੱਚ ਸਿਸਟਮ ਡਾਟਾ ਤੋਂ ਪੈਟਰਨ ਸਿੱਖ ਕੇ ਨਵੇਂ ਇਨਪੁਟ ਉੱਤੇ ਅਨੁਮਾਨ ਜਾਂ ਫੈਸਲਾ ਕਰਦਾ ਹੈ।",
        "verify": "ਜਵਾਬ ਪਤਾ ਨਾ ਹੋਵੇ ਤਾਂ ਕੀ ਕਰਨਾ ਚਾਹੀਦਾ ਹੈ?", "verify_r": "ਸਪਸ਼ਟ ਦੱਸੋ ਕਿ ਜਵਾਬ ਪਤਾ ਨਹੀਂ ਅਤੇ ਜਾਂਚ ਤੋਂ ਬਿਨਾਂ ਜਾਣਕਾਰੀ ਨਾ ਬਣਾਓ।",
        "translate": "ਅਨੁਵਾਦ ਵਿੱਚ ਗਿਣਤੀਆਂ ਕਿਵੇਂ ਰੱਖਣੀਆਂ ਚਾਹੀਦੀਆਂ ਹਨ?", "translate_r": "ਗਿਣਤੀਆਂ, ਨਾਮ ਅਤੇ ਤਰੀਖਾਂ ਬਿਲਕੁਲ ਸਹੀ ਰੱਖਣੀਆਂ ਚਾਹੀਦੀਆਂ ਹਨ।",
        "safety": "ਅਸੁਰੱਖਿਅਤ ਤਰੀਕਾ ਮੰਗਿਆ ਜਾਵੇ ਤਾਂ ਕੀ ਕਰਨਾ ਚਾਹੀਦਾ ਹੈ?", "safety_r": "ਅਸੁਰੱਖਿਅਤ ਤਰੀਕਾ ਨਾ ਦਿਓ ਅਤੇ ਸੁਰੱਖਿਅਤ ਵਿਕਲਪ ਦੱਸੋ।",
    },
    "or": {
        "add": "{a} + {b} କେତେ?", "add_r": "{a} + {b} = {r}.",
        "sub": "{a} ରୁ {b} ବିୟୋଗ କଲେ କେତେ?", "sub_r": "{a} - {b} = {r}.",
        "mul": "{a} କୁ {b} ଦ୍ୱାରା ଗୁଣ କଲେ କେତେ?", "mul_r": "{a} × {b} = {r}.",
        "pct": "{a} ର {b} ପ୍ରତିଶତ କେତେ?", "pct_r": "{a} ର {b}% = {r}.",
        "even": "{a} ଯୁଗ୍ମ ସଂଖ୍ୟା କି?", "even_r": "ହଁ, {a} ଯୁଗ୍ମ ସଂଖ୍ୟା ଅଟେ।",
        "odd": "{a} ବିଜୋଡ଼ ସଂଖ୍ୟା କି?", "odd_r": "ହଁ, {a} ବିଜୋଡ଼ ସଂଖ୍ୟା ଅଟେ।",
        "compare": "{a} ଏବଂ {b} ମଧ୍ୟରେ ବଡ଼ କେଉଁଟି?", "compare_r": "ବଡ଼ ସଂଖ୍ୟା {r}।",
        "clarify": "ଅସ୍ପଷ୍ଟ ପ୍ରଶ୍ନ ହେଲେ କଣ କରିବା ଉଚିତ?", "clarify_r": "ଆବଶ୍ୟକ ବିବରଣୀ ପଚାରନ୍ତୁ ଏବଂ ଜୋଖିମପୂର୍ଣ୍ଣ ଅନୁମାନ ନକରନ୍ତୁ।",
        "learn": "ମେସିନ ଲର୍ଣ୍ଣିଂ କଣ?", "learn_r": "ମେସିନ ଲର୍ଣ୍ଣିଂ ହେଉଛି ତଥ୍ୟରୁ ଧାରା ଶିଖି ନୂତନ ଇନପୁଟରେ ଅନୁମାନ କିମ୍ବା ନିଷ୍ପତ୍ତି କରିବା ପ୍ରକ୍ରିୟା।",
        "verify": "ଉତ୍ତର ଜଣା ନଥିଲେ କଣ କରିବା ଉଚିତ?", "verify_r": "ଜଣା ନାହିଁ ବୋଲି ସ୍ପଷ୍ଟ କହନ୍ତୁ ଏବଂ ଯାଞ୍ଚ ବିନା ସୂଚନା ତିଆରି କରନ୍ତୁ ନାହିଁ।",
        "translate": "ଅନୁବାଦରେ ସଂଖ୍ୟା କିପରି ରଖିବେ?", "translate_r": "ସଂଖ୍ୟା, ନାମ ଏବଂ ତାରିଖକୁ ସଠିକ ରଖିବା ଉଚିତ।",
        "safety": "ଅସୁରକ୍ଷିତ ପଦ୍ଧତି ପଚାରାଗଲେ କଣ କରିବେ?", "safety_r": "ଅସୁରକ୍ଷିତ ପଦ୍ଧତି ଦେବେ ନାହିଁ ଏବଂ ସୁରକ୍ଷିତ ବିକଳ୍ପ ଦେଖାନ୍ତୁ।",
    },
    "ur": {
        "add": "{a} + {b} کتنا ہے؟", "add_r": "{a} + {b} = {r}۔",
        "sub": "{a} میں سے {b} کم کریں تو کتنا ہوگا؟", "sub_r": "{a} - {b} = {r}۔",
        "mul": "{a} کو {b} سے ضرب دیں تو کتنا ہوگا؟", "mul_r": "{a} × {b} = {r}۔",
        "pct": "{a} کا {b} فیصد کتنا ہے؟", "pct_r": "{a} کا {b}% = {r}۔",
        "even": "کیا {a} جفت عدد ہے؟", "even_r": "جی ہاں، {a} ایک جفت عدد ہے۔",
        "odd": "کیا {a} طاق عدد ہے؟", "odd_r": "جی ہاں، {a} ایک طاق عدد ہے۔",
        "compare": "{a} اور {b} میں بڑا عدد کون سا ہے؟", "compare_r": "بڑا عدد {r} ہے۔",
        "clarify": "غیر واضح سوال پر کیا کرنا چاہیے؟", "clarify_r": "ضروری معلومات پوچھیں اور خطرناک اندازہ نہ لگائیں۔",
        "learn": "مشین لرننگ کیا ہے؟", "learn_r": "مشین لرننگ وہ طریقہ ہے جس میں نظام ڈیٹا سے نمونے سیکھ کر نئے ان پٹ پر اندازہ یا فیصلہ کرتا ہے۔",
        "verify": "جواب معلوم نہ ہو تو کیا کرنا چاہیے؟", "verify_r": "واضح طور پر کہیں کہ جواب معلوم نہیں اور بغیر جانچ کے معلومات نہ گھڑیں۔",
        "translate": "ترجمے میں نمبروں کو کیسے برقرار رکھنا چاہیے؟", "translate_r": "نمبروں، ناموں اور تاریخوں کو درست طور پر برقرار رکھنا چاہیے۔",
        "safety": "غیر محفوظ طریقہ مانگا جائے تو کیا کرنا چاہیے؟", "safety_r": "غیر محفوظ طریقہ فراہم نہ کریں اور محفوظ متبادل تجویز کریں۔",
    },
    "as": {
        "add": "{a} + {b} কিমান?", "add_r": "{a} + {b} = {r}।",
        "sub": "{a} ৰ পৰা {b} বিয়োগ কৰিলে কিমান?", "sub_r": "{a} - {b} = {r}।",
        "mul": "{a} ক {b} ৰে গুণ কৰিলে কিমান?", "mul_r": "{a} × {b} = {r}।",
        "pct": "{a} ৰ {b} শতাংশ কিমান?", "pct_r": "{a} ৰ {b}% = {r}।",
        "even": "{a} যুগ্ম সংখ্যা নেকি?", "even_r": "হয়, {a} এটা যুগ্ম সংখ্যা।",
        "odd": "{a} বিজোড় সংখ্যা নেকি?", "odd_r": "হয়, {a} এটা বিজোড় সংখ্যা।",
        "compare": "{a} আৰু {b}ৰ মাজত ডাঙৰ কোনটো?", "compare_r": "ডাঙৰ সংখ্যা {r}।",
        "clarify": "অস্পষ্ট প্ৰশ্ন হ'লে কি কৰিব লাগে?", "clarify_r": "প্ৰয়োজনীয় তথ্য সুধিব লাগে আৰু বিপদজনক অনুমান নকৰিব লাগে।",
        "learn": "মেচিন লাৰ্নিং কি?", "learn_r": "মেচিন লাৰ্নিং হৈছে তথ্যৰ পৰা ধৰণ শিকি নতুন ইনপুটত অনুমান বা সিদ্ধান্ত লোৱাৰ পদ্ধতি।",
        "verify": "উত্তৰ নাজানিলে কি কৰিব লাগে?", "verify_r": "নাজানো বুলি স্পষ্টকৈ কওক আৰু যাচাই নকৰাকৈ তথ্য নসজাব।",
        "translate": "অনুবাদত সংখ্যাবোৰ কেনেকৈ ৰাখিব লাগে?", "translate_r": "সংখ্যা, নাম আৰু তাৰিখ সঠিকভাৱে ৰাখিব লাগে।",
        "safety": "অসুৰক্ষিত পদ্ধতি বিচাৰিলে কি কৰিব লাগে?", "safety_r": "অসুৰক্ষিত পদ্ধতি নিদি সুৰক্ষিত বিকল্প আগবঢ়াব লাগে।",
    },
}


def _load(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("source") != SOURCE:
            rows.append(row)
    return rows


def _make_rows(per_template: int = 100) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for lang, t in LANGUAGES.items():
        for i in range(1, per_template + 1):
            a = 11 * i + 3
            b = 7 * i + 2
            values = {
                "a": a, "b": b, "r": a + b,
            }
            rows.append({"instruction": t["add"].format(**values), "response": t["add_r"].format(**values), "category": "math", "language": lang, "source": SOURCE})
            rows.append({"instruction": t["sub"].format(a=a + b, b=b), "response": t["sub_r"].format(a=a + b, b=b, r=a), "category": "math", "language": lang, "source": SOURCE})
            rows.append({"instruction": t["mul"].format(**values), "response": t["mul_r"].format(a=a, b=b, r=a * b), "category": "math", "language": lang, "source": SOURCE})
            pct = (i % 9) + 1
            pct_base = (i + 4) * 10
            rows.append({"instruction": t["pct"].format(a=pct_base, b=pct), "response": t["pct_r"].format(a=pct_base, b=pct, r=pct_base * pct / 100), "category": "math", "language": lang, "source": SOURCE})
            even_n = a if a % 2 == 0 else a + 1
            odd_n = b if b % 2 else b + 1
            rows.append({"instruction": t["even"].format(a=even_n), "response": t["even_r"].format(a=even_n), "category": "reasoning", "language": lang, "source": SOURCE})
            rows.append({"instruction": t["odd"].format(a=odd_n), "response": t["odd_r"].format(a=odd_n), "category": "reasoning", "language": lang, "source": SOURCE})
            bigger = max(a, b)
            rows.append({"instruction": t["compare"].format(a=a, b=b), "response": t["compare_r"].format(r=bigger), "category": "reasoning", "language": lang, "source": SOURCE})
            rows.append({"instruction": f"{t['clarify']} ({i})", "response": t["clarify_r"], "category": "conversation", "language": lang, "source": SOURCE})
            rows.append({"instruction": f"{t['learn']} ({i})", "response": t["learn_r"], "category": "education", "language": lang, "source": SOURCE})
            rows.append({"instruction": f"{t['verify']} ({i})", "response": t["verify_r"], "category": "honesty", "language": lang, "source": SOURCE})
            rows.append({"instruction": f"{t['translate']} ({i})", "response": t["translate_r"], "category": "translation", "language": lang, "source": SOURCE})
            rows.append({"instruction": f"{t['safety']} ({i})", "response": t["safety_r"], "category": "safety", "language": lang, "source": SOURCE})
    return rows


def build() -> dict[str, int]:
    instructions = _load(INSTRUCTION_PATH)
    multilingual = _load(MULTILINGUAL_PATH)
    generated = _make_rows()
    instructions.extend(generated)
    multilingual.extend(generated)

    INSTRUCTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    INSTRUCTION_PATH.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in instructions), encoding="utf-8")
    MULTILINGUAL_PATH.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in multilingual), encoding="utf-8")

    corpus_seed = "\n".join(f"{row['instruction']} {row['response']}" for row in generated)
    repeats = max(1, (1_000_000 // max(1, len(corpus_seed))) + 1)
    corpus = (corpus_seed + "\n") * repeats
    CORPUS_PATH.write_text(corpus[:1_100_000], encoding="utf-8")
    return {"generated_rows": len(generated), "instruction_rows": len(instructions), "multilingual_rows": len(multilingual), "corpus_chars": min(len(corpus), 1_100_000)}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, ensure_ascii=False))
