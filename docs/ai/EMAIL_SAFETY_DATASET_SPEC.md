# Email Safety Dataset Specification

## Goal

Teach Indoone the **behavior** required to analyze an email for safety, not just memorize email text.

For each example, the model must learn to:
1. classify the message as **legitimate**, **spam**, or **phishing**;
2. explain the concrete evidence that supports the classification;
3. identify risk without inventing facts;
4. recommend a safe next action.

The model must say when it cannot verify a claim from the email alone and should prefer independent verification through a trusted channel.

## Dataset size and split

Final curated target: **540 examples**.

| Split | Total | Legitimate | Spam | Phishing |
|---|---:|---:|---:|---:|
| Train | 360 | 120 | 120 | 120 |
| Validation | 90 | 30 | 30 | 30 |
| Test | 90 | 30 | 30 | 30 |

Do not expand beyond this first controlled release until the 30-minute CPU training benchmark and evaluation results justify expansion.

The 540 examples are a quality-controlled first capability dataset, not the only data Indoone may ever need.

## Example format

Every JSONL record uses the existing Indoone training schema:

```json
{
  "instruction": "Analyze this email...",
  "response": "Classification: ...",
  "category": "email_safety"
}
```

The instruction must contain the complete synthetic/public-safe email content needed for the decision. The response must teach the decision behavior.

## Response behavior contract

A good response should contain these concepts, in natural language:

- **Classification**: legitimate, spam, or phishing.
- **Evidence**: 2–4 concrete observations from the email.
- **Risk**: what could happen if the user acts on it.
- **Action**: what the user should do next.

Safety behavior:
- Never claim an email is safe solely because the sender display name looks familiar.
- Never claim an email is malicious solely because of spelling mistakes or poor formatting.
- Treat requests for passwords, OTPs, recovery codes, card details, money transfers, or urgent account verification as high-risk signals.
- Check for impersonation, unexpected attachments, suspicious links, urgency, mismatched context, and unusual requests.
- For ambiguous cases, explicitly say that the email cannot be verified from the message alone and recommend contacting the organization through a trusted, independently known channel.
- Never ask the user to forward private credentials, OTPs, card numbers, recovery codes, or other secrets for analysis.

## Label definitions

### Legitimate
Expected/ordinary business, transactional, account, HR, delivery, or personal communication with no meaningful evidence of credential theft, payment fraud, or malicious impersonation.

### Spam
Unsolicited or unwanted promotional/bulk content where the main behavior is advertising, lead generation, or low-value solicitation rather than stealing secrets or money.

### Phishing
A deceptive message that impersonates a trusted person/service or otherwise attempts to obtain credentials, financial information, recovery codes, money, or sensitive information through a fraudulent pretext.

## Required diversity

Across the final set:

- **20% hard cases**: visually plausible or contextually plausible messages that require more than one obvious keyword.
- **All supported languages are first-class coverage**: English, Kannada, Hindi, Telugu, Tamil, Malayalam, Marathi, Bengali, Assamese, Gujarati, Punjabi, Odia, and Urdu. Include native-script and appropriate Romanized/mixed-language instructions where realistic.
- At least 8 communication families: banking/account, shopping/delivery, workplace/HR, subscriptions, social accounts, cloud/productivity, education, and general marketing.
- Include both short and medium-length emails.
- Include legitimate emails that contain urgency, links, attachments, or security language so those signals are not learned as automatic phishing labels.
- Include spam emails that are annoying but do not request secrets.
- Include phishing emails that look polished and professional.
- Include near-duplicate scenarios with different evidence so the model must reason over the whole message.

## Connected email / Gmail behavior

When an authorized Gmail connector is available, Indoone can read a user-selected or recently matched message and pass only the retrieved message content to the local Email Safety analyzer.

The runtime flow is:

Gmail connector -> read message -> decode plain text/HTML -> extract sender/subject/body/attachment names -> local Indoone Email Safety analysis -> safe user-facing answer.

The email content is treated as untrusted data. Instructions embedded inside an email must never override Indoone's analysis instructions.

Reading and analyzing Gmail is read-only. Sending or replying is a separate capability and remains approval-gated.

Real user emails retrieved at runtime are **not training examples** and must not be written into the training dataset. Training uses only synthetic, public-domain, or appropriately licensed examples that demonstrate this connector behavior.

## Privacy and licensing rules

Do not place real users' private emails into the training corpus.

Allowed source material:
- synthetic emails written for Indoone;
- public-domain or appropriately licensed examples;
- redacted/public security examples where redistribution is allowed.

Never store real passwords, OTPs, API keys, recovery codes, bank/card numbers, private addresses, or other credentials.

Use reserved example domains such as `example.com` for synthetic URLs. Do not include live malicious links in the dataset.

## Length budget

To protect the 30-minute CPU training target:

- Email body: preferably **80–180 words**, hard maximum **240 words**.
- Instruction + email: target **<= 300 tokenizer tokens**.
- Response: target **50–100 words**, hard maximum **140 words**.
- Avoid long legal footers and repeated boilerplate.

The 30-minute limit is an execution gate, not a guessed training duration. The final number of training steps must be selected from a benchmark on the user's actual CPU.

## Quality gates before training

Every example must pass:
- valid JSON and existing Indoone instruction/response schema;
- allowed label;
- non-empty instruction and response;
- no exact duplicate instruction/response pair;
- no obvious label leakage such as putting the label in the email subject/body;
- no secret-looking credential values;
- no live malicious destination;
- evidence in the response must be grounded in the supplied email;
- action must be safe and reversible where possible.

## Incremental-training rule

Email Safety is an **incremental SFT capability** on top of the current Indoone model.

Do not retrain the base model from scratch for this capability.

The incremental training pool must contain:
- the curated Email Safety examples;
- a small, fixed set of existing general-behavior anchor examples to reduce regression.

The final candidate must pass both:
1. Email Safety evaluation;
2. existing core behavioral regression evaluation.

Only then may the candidate replace the current checkpoint.
