import pandas as pd
import random

# -------------------------
# SCAM TEMPLATES (ROJAK STYLE)
# -------------------------
scam_templates = [
    "Bank kau kena block, click link ni cepat {link}",
    "Account anda akan dibekukan segera, verify sekarang {link}",
    "Shopee parcel stuck kastam, bayar RM{amount} untuk release",
    "We detected login from {country}, confirm identity {link}",
    "You win RM{amount} cash, claim now sebelum expired",
    "OTP needed to verify transaction, send cepat",
    "TNG eWallet kena suspend, login untuk aktifkan {link}",
    "Bank Negara report: account suspicious activity, verify {link}",
    "Job part time RM{amount}/day, no experience needed, join link {link}",
    "Loan approved instant RM{amount}, send IC & bank details",
    "Police report: you involved in case, contact urgent {link}",
    "Investment crypto profit 200% guaranteed, join now {link}",
    "Your email hacked, reset password immediately {link}",
    "Free RM{amount} voucher, claim sekarang {link}",
    "Customs hold parcel, pay RM{amount} release fee",
]

# -------------------------
# SAFE TEMPLATES (ROJAK STYLE)
# -------------------------
safe_templates = [
    "Weh kau free tak malam ni?",
    "Assignment dah siap belum?",
    "Jom makan dekat cafe kampus",
    "Aku lambat sikit traffic jam",
    "Thanks bro tolong aku tadi",
    "Kita jumpa esok kelas 10am",
    "Apa plan weekend ni?",
    "Aku dah hantar report tu",
    "Good morning bro, bangun cepat",
    "You datang lecture tak hari ni?",
    "Let’s study together for exam",
    "Aku tengah buat project AI ni",
    "Where are you now bro?",
    "Dinner rumah mak malam ni",
    "Call aku nanti free",
]

countries = ["Nigeria", "India", "Philippines", "Unknown"]
domains = ["bit.ly/verify", "bank-secure.com/login", "sh0pee-help.net", "tng-update.com"]

data = []

# -------------------------
# GENERATE SCAM (1500)
# -------------------------
for _ in range(1500):
    template = random.choice(scam_templates)
    message = template.format(
        amount=random.randint(50, 5000),
        link=random.choice(domains),
        country=random.choice(countries)
    )
    data.append([message, "SCAM"])

# -------------------------
# GENERATE SAFE (1500)
# -------------------------
for _ in range(1500):
    message = random.choice(safe_templates)
    data.append([message, "SAFE"])

# Shuffle dataset
random.shuffle(data)

# Save CSV
df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv("malaysia_scam_dataset_3000.csv", index=False)

print("Dataset generated successfully!")
print(df.head())