import json
import random
from pathlib import Path

random.seed(9090)

LANGS = [
    ("English", "Analyze this email and classify it as legitimate, spam, or phishing. Give 2-4 grounded reasons and a safe next step."),
    ("Kannada", "ಈ email ಅನ್ನು ನೋಡಿ legitimate, spam ಅಥವಾ phishing ಎಂದು classify ಮಾಡಿ. 2-4 ನಿಜವಾದ ಕಾರಣಗಳು ಮತ್ತು safe next step ಹೇಳಿ."),
    ("Hindi", "इस email को देखें और legitimate, spam या phishing में classify करें। 2-4 ठोस कारण और सुरक्षित अगला कदम बताएं।"),
    ("Telugu", "ఈ email ను పరిశీలించి legitimate, spam లేదా phishing గా classify చేయండి. 2-4 స్పష్టమైన కారణాలు మరియు safe next step చెప్పండి."),
    ("Tamil", "இந்த email-ஐ பார்த்து legitimate, spam அல்லது phishing என்று classify செய்யுங்கள். 2-4 காரணங்களும் பாதுகாப்பான அடுத்த படியும் கூறுங்கள்."),
    ("Malayalam", "ഈ email പരിശോധിച്ച് legitimate, spam അല്ലെങ്കിൽ phishing എന്ന് classify ചെയ്യുക. 2-4 വ്യക്തമായ കാരണങ്ങളും സുരക്ഷിതമായ അടുത്ത നടപടിയും പറയുക."),
    ("Marathi", "हा email तपासा आणि legitimate, spam किंवा phishing म्हणून classify करा. 2-4 ठोस कारणे आणि सुरक्षित पुढची कृती सांगा."),
    ("Bengali", "এই email পরীক্ষা করে legitimate, spam নাকি phishing তা classify করুন। 2-4টি নির্দিষ্ট কারণ ও নিরাপদ পরবর্তী পদক্ষেপ বলুন."),
    ("Gujarati", "આ email તપાસીને legitimate, spam કે phishing તરીકે classify કરો. 2-4 ચોક્કસ કારણો અને સુરક્ષિત આગળનું પગલું આપો."),
    ("Punjabi", "ਇਸ email ਦੀ ਜਾਂਚ ਕਰਕੇ legitimate, spam ਜਾਂ phishing ਵਜੋਂ classify ਕਰੋ। 2-4 ਠੋਸ ਕਾਰਨ ਅਤੇ ਸੁਰੱਖਿਅਤ ਅਗਲਾ ਕਦਮ ਦਿਓ."),
    ("Odia", "ଏହି email ଯାଞ୍ଚ କରି legitimate, spam କି phishing ବୋଲି classify କରନ୍ତୁ। 2-4ଟି ସ୍ପଷ୍ଟ କାରଣ ଏବଂ ସୁରକ୍ଷିତ ପରବର୍ତ୍ତୀ ପଦକ୍ଷେପ ଦିଅନ୍ତୁ."),
    ("Urdu", "اس email کو دیکھ کر legitimate، spam یا phishing کے طور پر classify کریں۔ 2-4 واضح وجوہات اور محفوظ اگلا قدم بتائیں."),
    ("Mixed", "Ee email check maadi: legitimate, spam athva phishing? 2-4 grounded reasons + safe next step kodi."),
]

FAMILIES = [
    "banking/account", "shopping/delivery", "workplace/HR", "subscription/billing",
    "social account", "cloud/productivity", "education", "travel", "healthcare",
    "government/service", "job/recruitment", "general marketing",
]
NAMES = [
    "Example Bank", "Example Store", "Example Workspace", "Example Company",
    "Example Learning", "Example Travel", "Example Service", "Example Community",
]
SUBJECTS = {
    "legitimate": [
        "Monthly statement is ready", "Order shipment update", "Interview schedule confirmation",
        "Subscription invoice available", "New sign-in notice", "Shared document notification",
        "Course timetable updated", "Booking confirmation", "Appointment reminder",
        "Service request update", "Application status update", "Membership newsletter",
    ],
    "spam": [
        "Special offer just for you", "Limited time reward", "Exclusive discount inside",
        "Free gift waiting", "Weekend deal", "Member-only promotion",
        "You have been selected", "Bonus offer available", "Save more this week",
    ],
    "phishing": [
        "Urgent: verify your account", "Action required: unusual activity",
        "Your account will be suspended", "Security verification needed",
        "Payment failed — confirm now", "Reset required immediately",
        "Compliance notice — verify identity", "Unusual sign-in — secure your account",
    ],
}
BODIES = {
    "legitimate": [
        "This message confirms a routine event related to an account, transaction, delivery, schedule, or service. It provides normal context and directs the recipient to the service's usual app or portal. It does not ask for a password, OTP, recovery code, payment card secret, or unusual transfer. The recipient can review the details normally and use an independently known support channel if anything looks unexpected.",
        "This is a routine business notification. It explains an expected update and includes ordinary reference information. No sensitive credential is requested, no payment is demanded, and the message does not pressure the recipient to bypass normal account security. For an important action, the recipient can open the organization's official app or website independently rather than relying on links in the message.",
    ],
    "spam": [
        "This message promotes an unsolicited product, discount, reward, or marketing opportunity. It uses attention-grabbing language and may encourage a quick visit to a promotional page, but it does not primarily attempt to steal credentials or impersonate a trusted security service. The recipient should treat unexpected offers cautiously and can ignore, delete, or report the message if it is unwanted.",
        "This is an unsolicited promotional message with a strong sales or reward angle. The offer may be exaggerated or irrelevant to the recipient. There is no clear evidence in the message of a credential-theft pretext, but unexpected promotional links or attachments still deserve caution. The safest response is to avoid interacting with the offer unless it can be independently verified.",
    ],
    "phishing": [
        "This message claims there is an urgent account, payment, security, or identity problem and pushes the recipient to act through the email. The request may imitate a trusted organization and create fear about suspension or loss of access. Do not use the embedded action to sign in or provide secrets. Verify the situation through the organization's official app or website opened independently.",
        "The email uses a deceptive pretext such as unusual activity, a failed payment, or an immediate security check. It asks the recipient to verify or secure something and may direct them to an attacker-controlled page. The key risk is credential or financial theft. Do not provide passwords, OTPs, recovery codes, card details, or API secrets; verify independently through a trusted channel.",
    ],
}
HARD_CUES = [
    "The sender display name looks familiar, but the content must be judged on the actual request and context.",
    "The wording is polished, so formatting quality alone is not evidence that the message is safe.",
    "The message contains security-related language, but that signal must be weighed with the actual requested action.",
    "The email includes an attachment name, but the attachment is not proof of legitimacy or maliciousness by itself.",
    "The message creates some urgency, so the recipient should distinguish a genuine deadline from pressure to bypass normal verification.",
]
REASONS = {
    "legitimate": [
        "It describes a routine service or transaction event.",
        "It does not request passwords, OTPs, recovery codes, or card secrets.",
        "It does not force the recipient to bypass normal verification.",
        "Independent verification through the usual app or portal is available.",
    ],
    "spam": [
        "It is unsolicited promotional or reward-oriented content.",
        "The main goal is advertising rather than credential theft.",
        "The offer may be exaggerated or irrelevant to the recipient.",
        "Unexpected promotional links should still be treated cautiously.",
    ],
    "phishing": [
        "It uses urgency or fear around an account, payment, or security issue.",
        "It pushes the recipient toward an email-driven verification action.",
        "It may impersonate a trusted organization or service.",
        "The safest option is independent verification instead of using the email action.",
    ],
}
ACTIONS = {
    "legitimate": "Safe next step: review normally; for an important action, open the official app or website yourself.",
    "spam": "Safe next step: do not engage with the unsolicited offer; delete or report it if unwanted.",
    "phishing": "Safe next step: do not click the email link or share secrets; verify through the official app or website.",
}

COMPOSE = [
    "Write a polite leave request to my manager for Friday.",
    "Compose a professional email asking a customer for a convenient meeting time.",
    "Write an email to support asking for an update on my existing ticket.",
    "Compose a short thank-you email after an interview.",
    "Write an email requesting a copy of an invoice.",
    "Compose a professional email asking a teacher for the assignment deadline.",
    "Write a concise email to reschedule a meeting to next week.",
    "Compose an email asking a service team to explain a billing charge.",
    "Write a polite follow-up about a previous application.",
    "Compose an email notifying a teammate that the requested document is attached.",
]
REPLY = [
    "Reply politely to a meeting request and suggest 3 PM tomorrow.",
    "Reply to a customer saying the issue is being reviewed and you will update them soon.",
    "Reply to a manager confirming that the requested document will be shared today.",
    "Reply to an interview invitation and say you are available on Tuesday afternoon.",
    "Reply to a support email asking for the ticket reference before proceeding.",
    "Reply to a delivery email saying the recipient will be available tomorrow.",
    "Reply to a colleague thanking them for the information.",
    "Reply to a school email confirming receipt of the notice.",
    "Reply to a billing email asking for clarification on the charge.",
    "Reply to a project update acknowledging the message and next step.",
]
ACTION_LANGS = ["English", "Kannada", "Hindi", "Telugu", "Tamil", "Malayalam", "Marathi", "Bengali", "Gujarati", "Punjabi", "Odia", "Urdu", "Mixed"]

def action_prompt(base, lang):
    wrappers = {
        "English": base,
        "Kannada": "Email request: " + base + " ಇದನ್ನು professional ಆಗಿ ಬರೆಯಿರಿ.",
        "Hindi": "Email request: " + base + " इसे professional तरीके से लिखें।",
        "Telugu": "Email request: " + base + " దీన్ని professional గా రాయండి.",
        "Tamil": "Email request: " + base + " இதை professional ஆக எழுதுங்கள்.",
        "Malayalam": "Email request: " + base + " ഇത് professional ആയി തയ്യാറാക്കുക.",
        "Marathi": "Email request: " + base + " हे professional पद्धतीने लिहा.",
        "Bengali": "Email request: " + base + " এটি professional ভাবে লিখুন.",
        "Gujarati": "Email request: " + base + " આને professional રીતે લખો.",
        "Punjabi": "Email request: " + base + " ਇਸਨੂੰ professional ਢੰਗ ਨਾਲ ਲਿਖੋ.",
        "Odia": "Email request: " + base + " ଏହାକୁ professional ଭାବେ ଲେଖନ୍ତୁ.",
        "Urdu": "Email request: " + base + " اسے professional انداز میں لکھیں۔",
        "Mixed": "Mail request: " + base + " professional style alli compose maadi.",
    }
    return wrappers[lang]

def build():
    safety = []
    for label in ("legitimate", "spam", "phishing"):
        for i in range(160):
            lang, prompt = random.choice(LANGS)
            subject = random.choice(SUBJECTS[label])
            family = random.choice(FAMILIES)
            sender = random.choice(NAMES)
            body = random.choice(BODIES[label]) + " " + random.choice(HARD_CUES)
            body += " " + [
                "Reference: 4821.", "Reference: 7315.", "Update window: this week.",
                "The recipient did not request any special exception.",
                "No account credentials are included in the message.",
            ][i % 5]
            instruction = (
                f"{prompt}\n\nEMAIL:\nSubject: {subject}\n"
                f"From: {sender} <notice@example.com>\nTo: User <user@example.com>\n\n{body}"
            )
            reasons = random.sample(REASONS[label], 3)
            response = f"Classification: {label}. " + " ".join(reasons) + " " + ACTIONS[label]
            safety.append({"instruction": instruction, "response": response, "category": "email_safety"})
    random.shuffle(safety)

    actions = []
    for i in range(30):
        lang = random.choice(ACTION_LANGS)
        actions.append({
            "instruction": action_prompt(random.choice(COMPOSE), lang) + f" Variant {i+1}.",
            "response": "Compose the requested email without sending it. Use only recipient details explicitly supplied by the user; do not invent an address. Return a clear subject and body.",
            "category": "email_compose",
        })
        lang = random.choice(ACTION_LANGS)
        actions.append({
            "instruction": action_prompt(random.choice(REPLY), lang) + f" Variant {i+1}.",
            "response": "Prepare the reply for the existing email thread and preserve the thread context. Do not send it yet. Show the reply and require explicit user confirmation before sending.",
            "category": "email_reply",
        })
        lang = random.choice(ACTION_LANGS)
        send_base = [
            "The email is ready. Send it now only after I explicitly confirm.",
            "I reviewed the message. Ask for my confirmation before sending.",
            "The reply is prepared. Do not perform the send action until I confirm.",
        ][i % 3]
        actions.append({
            "instruction": action_prompt(send_base, lang) + f" Variant {i+1}.",
            "response": "Sending is an external action. Require explicit user confirmation immediately before the send operation. Do not send based only on the earlier request.",
            "category": "email_send",
        })
    random.shuffle(actions)
    return safety, actions

if __name__ == "__main__":
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)
    safety, actions = build()
    (out / "indoone_email_safety_additional_generated.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in safety),
        encoding="utf-8",
    )
    (out / "indoone_email_actions_additional_generated.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in actions),
        encoding="utf-8",
    )
    print(f"safety_additional={len(safety)}")
    print(f"actions_additional={len(actions)}")
