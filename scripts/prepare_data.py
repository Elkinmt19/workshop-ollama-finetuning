#!/usr/bin/env python3
"""Prepare and validate training data for fine-tuning."""

import argparse
import json
import logging
import random
from pathlib import Path

from src.data_loader import DataLoader, TrainingExample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Parameterized generators for unique synthetic examples ---

MERCHANTS_BY_CATEGORY = {
    "Food & Delivery": [
        ("RAPPI*RAPPI COL", 25_000, 85_000),
        ("IFOOD COL", 20_000, 70_000),
        ("UBER EATS", 22_000, 65_000),
        ("DOMINOS PIZZA", 30_000, 95_000),
        ("MCDONALDS APP", 15_000, 55_000),
        ("WOK", 35_000, 75_000),
        ("CREPES & WAFFLES", 28_000, 90_000),
        ("FRISBY", 18_000, 50_000),
        ("SUBWAY", 15_000, 38_000),
        ("JUAN VALDEZ CAFE", 8_000, 28_000),
    ],
    "Groceries & Supermarket": [
        ("EXITO", 50_000, 350_000),
        ("CARULLA", 60_000, 280_000),
        ("JUMBO", 80_000, 400_000),
        ("D1", 20_000, 120_000),
        ("ARA", 25_000, 150_000),
        ("OLIMPICA", 40_000, 250_000),
        ("MAKRO", 100_000, 500_000),
        ("MERCADO LIBRE SUPERMERCADO", 30_000, 200_000),
    ],
    "Transportation": [
        ("UBER *TRIP", 8_000, 35_000),
        ("DIDI *VIAJE", 7_000, 30_000),
        ("INDRIVER", 6_000, 28_000),
        ("BEAT *TRIP", 8_000, 32_000),
        ("PEAJE AUTOPISTA NORTE", 10_000, 16_000),
        ("PEAJE RUTA DEL SOL", 12_000, 22_000),
        ("TERPEL GASOLINA", 50_000, 180_000),
        ("PARQUEADERO CENTRO", 5_000, 15_000),
    ],
    "Entertainment & Subscriptions": [
        ("NETFLIX.COM", 32_000, 55_000),
        ("SPOTIFY", 14_900, 29_000),
        ("YOUTUBE PREMIUM", 18_900, 23_900),
        ("DISNEY+", 25_900, 33_900),
        ("HBO MAX", 17_900, 27_900),
        ("AMAZON PRIME", 24_900, 34_900),
        ("APPLE.COM/BILL", 5_900, 49_900),
        ("XBOX GAME PASS", 32_900, 54_900),
        ("PARAMOUNT+", 14_900, 22_900),
    ],
    "Utilities & Bills": [
        ("PSE ENEL CODENSA", 80_000, 220_000),
        ("PSE ETB", 60_000, 120_000),
        ("PSE CLARO MOVIL", 40_000, 95_000),
        ("PSE ACUEDUCTO BOGOTA", 50_000, 150_000),
        ("PSE GAS NATURAL", 30_000, 80_000),
        ("PSE TIGO", 45_000, 110_000),
        ("PSE MOVISTAR", 50_000, 100_000),
    ],
    "Health & Pharmacy": [
        ("DROGUERIA OLIMPICA", 15_000, 80_000),
        ("FARMATODO", 10_000, 120_000),
        ("CRUZ VERDE", 12_000, 95_000),
        ("LOCATEL", 20_000, 200_000),
        ("COPAGOS EPS SURA", 10_000, 50_000),
        ("COLMEDICA CITA", 15_000, 80_000),
    ],
    "Shopping & Clothing": [
        ("FALABELLA.COM.CO", 80_000, 500_000),
        ("ZARA COLOMBIA", 100_000, 350_000),
        ("CENTRO COMERCIAL", 50_000, 400_000),
        ("MERCADOLIBRE", 30_000, 600_000),
        ("ALKOSTO", 100_000, 800_000),
        ("H&M COLOMBIA", 60_000, 280_000),
        ("PULL AND BEAR", 70_000, 250_000),
    ],
    "Transfers & Payments": [
        ("TRANSFERENCIA NEQUI", 50_000, 2_000_000),
        ("TRANSFERENCIA BANCOLOMBIA", 100_000, 3_000_000),
        ("TRANSFERENCIA DAVIPLATA", 30_000, 1_000_000),
        ("PAGO PSE ARRIENDO", 800_000, 2_500_000),
    ],
    "Education": [
        ("COURSERA INC", 100_000, 200_000),
        ("UDEMY", 40_000, 150_000),
        ("PLATZI", 60_000, 90_000),
        ("ICETEX CUOTA", 200_000, 600_000),
        ("UNIVERSIDAD ANDES MATRICULA", 2_000_000, 8_000_000),
    ],
    "Insurance": [
        ("SEGUROS SURA POLIZA", 100_000, 400_000),
        ("SEGUROS BOLIVAR", 80_000, 350_000),
        ("MAPFRE COLOMBIA", 90_000, 300_000),
        ("LIBERTY SEGUROS", 85_000, 280_000),
    ],
}

CATEGORY_DESCRIPTIONS = {
    "Food & Delivery": "food delivery or restaurant purchase",
    "Groceries & Supermarket": "grocery or supermarket purchase",
    "Transportation": "transportation or mobility expense",
    "Entertainment & Subscriptions": "entertainment subscription or digital service",
    "Utilities & Bills": "utility bill payment via PSE online banking",
    "Health & Pharmacy": "health-related purchase or medical payment",
    "Shopping & Clothing": "retail or online shopping purchase",
    "Transfers & Payments": "person-to-person or bill transfer",
    "Education": "education-related payment or subscription",
    "Insurance": "insurance premium payment",
}

LOCATIONS = [
    "CALLE 80", "CHAPINERO", "USAQUEN", "SUBA", "KENNEDY",
    "CEDRITOS", "ZONA T", "CHICO", "CENTRO", "BOSA",
    "FONTIBÓN", "ENGATIVÁ", "TEUSAQUILLO", "SANTA FE",
]

HOURS_BY_CONTEXT = {
    "normal": (7, 22),
    "late_night": (0, 5),
    "early_morning": (5, 8),
    "business": (8, 18),
}

CUSTOMER_PROFILES = [
    {"desc": "office worker, stable income $4,500,000/month, moderate spender", "avg_monthly": 3_200_000},
    {"desc": "university student, income $1,200,000/month from part-time job", "avg_monthly": 900_000},
    {"desc": "freelance designer, variable income $3,000,000-$7,000,000/month", "avg_monthly": 4_500_000},
    {"desc": "retired teacher, pension $2,800,000/month, conservative spender", "avg_monthly": 1_800_000},
    {"desc": "young professional, income $6,500,000/month, high lifestyle spending", "avg_monthly": 5_500_000},
    {"desc": "small business owner, mixed personal/business spending, income $8,000,000/month", "avg_monthly": 6_000_000},
    {"desc": "single parent, income $3,500,000/month, budget-conscious", "avg_monthly": 3_000_000},
    {"desc": "recent graduate, first job $2,500,000/month, building financial habits", "avg_monthly": 2_200_000},
]

FRAUD_REASONS_HIGH = [
    "Unusual time — outside normal activity hours",
    "Amount significantly exceeds typical spending pattern",
    "Unknown merchant not in purchase history",
    "Geographic impossibility — activity in two distant cities",
    "Rapid succession of transactions (velocity anomaly)",
    "New device + password change before large transfer",
    "Amount just below reporting threshold (structuring)",
    "First-ever transaction in this category",
]

FRAUD_REASONS_LOW = [
    "Known merchant the customer frequents regularly",
    "Amount within established spending range",
    "Time of day consistent with typical activity pattern",
    "Category matches historical behavior",
    "Location matches customer's home city",
    "Regular recurring payment at expected interval",
]


def _random_date(rng: random.Random) -> str:
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    return f"2024-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"


def _format_amount(amount: int) -> str:
    return f"${amount:,}".replace(",", ",") + " COP"


def _generate_categorization(rng: random.Random) -> dict:
    category = rng.choice(list(MERCHANTS_BY_CATEGORY.keys()))
    merchants = MERCHANTS_BY_CATEGORY[category]
    merchant_name, min_amt, max_amt = rng.choice(merchants)
    amount = rng.randint(min_amt // 1000, max_amt // 1000) * 1000
    location = rng.choice(LOCATIONS) if rng.random() > 0.5 else ""
    full_merchant = f"{merchant_name} {location}".strip() if location else merchant_name
    date = _random_date(rng)
    desc = CATEGORY_DESCRIPTIONS[category]

    return {
        "instruction": "Categorize the following bank transaction based on its description.",
        "input": f"{full_merchant} - {_format_amount(amount)} - {date}",
        "output": f"Category: {category}. This is a {desc}. Amount of {_format_amount(amount)} charged on {date}.",
    }


def _generate_fraud_analysis(rng: random.Random) -> dict:
    is_fraud = rng.random() > 0.4  # 60% high risk, 40% low risk
    profile = rng.choice(CUSTOMER_PROFILES)
    category = rng.choice(list(MERCHANTS_BY_CATEGORY.keys()))
    merchants = MERCHANTS_BY_CATEGORY[category]
    merchant_name, min_amt, max_amt = rng.choice(merchants)

    if is_fraud:
        # Make the amount suspicious (2-5x the profile average for a single tx)
        amount = rng.randint(profile["avg_monthly"], profile["avg_monthly"] * 3)
        hour = rng.randint(0, 5)  # Late night
        minute = rng.randint(0, 59)
        date = f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d} {hour:02d}:{minute:02d}"
        reasons = rng.sample(FRAUD_REASONS_HIGH, k=rng.randint(2, 3))
        reasons_text = "; ".join(f"({i+1}) {r}" for i, r in enumerate(reasons))
        risk = "HIGH RISK"
        recommendation = "Recommendation: Flag for verification and send real-time alert to customer."
    else:
        amount = rng.randint(min_amt, max_amt)
        hour = rng.randint(8, 20)
        minute = rng.randint(0, 59)
        date = f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d} {hour:02d}:{minute:02d}"
        reasons = rng.sample(FRAUD_REASONS_LOW, k=rng.randint(2, 3))
        reasons_text = "; ".join(f"({i+1}) {r}" for i, r in enumerate(reasons))
        risk = "LOW RISK"
        recommendation = "No action needed."

    return {
        "instruction": "Analyze this transaction and determine if it shows signs of potential fraud. Explain your reasoning.",
        "input": f"Transaction: {merchant_name} - {_format_amount(amount)} - {date}. Customer profile: {profile['desc']}, average monthly spending {_format_amount(profile['avg_monthly'])}.",
        "output": f"{risk} - {reasons_text}. {recommendation}",
    }


def _generate_spending_insight(rng: random.Random) -> dict:
    categories = rng.sample(list(MERCHANTS_BY_CATEGORY.keys()), k=rng.randint(5, 7))
    total = rng.randint(2_000_000, 8_000_000)
    remaining = total
    breakdown = []
    for i, cat in enumerate(categories):
        if i == len(categories) - 1:
            amount = remaining
        else:
            amount = rng.randint(remaining // (len(categories) - i + 1),
                                remaining // max(1, len(categories) - i - 1))
            amount = min(amount, remaining - (len(categories) - i - 1) * 50_000)
        remaining -= amount
        pct = round(amount / total * 100)
        breakdown.append(f"{cat}: {_format_amount(amount)} ({pct}%)")

    breakdown_text = ", ".join(breakdown)
    highest_cat = categories[0]
    highest_pct = round((total - remaining + breakdown[0].split("(")[0].count(",")) / total * 100)

    insights = [
        f"Your highest spending category this month is {categories[0]}.",
        f"Consider setting category budgets to track spending against targets.",
        f"Total monthly spending of {_format_amount(total)} {'exceeds' if total > 4_000_000 else 'is within'} the recommended 80% of typical income for your profile.",
    ]

    return {
        "instruction": "Analyze the following monthly transaction summary and provide spending insights and recommendations.",
        "input": f"Monthly Summary: {breakdown_text}. Total: {_format_amount(total)}.",
        "output": f"Key Insights: {' '.join(insights)} Recommendation: Review your top 2 categories for optimization opportunities and set up automatic savings of at least 15% of income.",
    }


def _generate_customer_qa(rng: random.Random) -> dict:
    qa_templates = [
        {
            "q": "I see a charge of {amount} from {merchant} but I don't remember making this purchase. What should I do?",
            "a": "Let's investigate this unrecognized charge: (1) Check the date and time — were you near a store or online shopping then?; (2) Check if a family member or authorized user made the purchase; (3) Look for an email confirmation from {merchant}; (4) If still unrecognized after checking, you can dispute it through the app under Transactions > Select charge > Report Problem. We'll issue a provisional credit within 5 business days while investigating.",
        },
        {
            "q": "My transfer of {amount} to {recipient} has been 'processing' for 2 days. Is this normal?",
            "a": "Transfer processing times vary: (1) Nequi-to-Nequi transfers are instant — if it's processing, there may be a system delay; (2) Transfers to other banks via ACH take 1-2 business days (excluding weekends/holidays); (3) Large amounts over $10,000,000 may require additional verification; (4) Check if the recipient's bank details are correct. If it's been over 48 business hours, contact support and we'll trace the transfer with reference number.",
        },
        {
            "q": "Why do I see a charge labeled 'IVA SERVICIOS FINANCIEROS' of {amount}? I didn't authorize this.",
            "a": "The 'IVA Servicios Financieros' charge is a government-mandated tax (19% VAT) applied to certain financial services. Common triggers: (1) Monthly account fees if applicable to your plan; (2) International transaction fees (the IVA is on the fee, not the purchase); (3) Credit card annual fee (IVA on the fee amount); (4) Insurance premiums linked to your account. Check which service the IVA applies to in your transaction detail view. This is a regulatory charge, not an error.",
        },
        {
            "q": "I withdrew {amount} from an ATM but received less cash. The machine gave me the wrong amount.",
            "a": "ATM discrepancy claims need immediate action: (1) DO NOT leave the ATM — check if the machine displays an error or printed a receipt; (2) Note the exact ATM location, time, and the amount you actually received vs charged; (3) Report immediately through the app: Help > ATM Issues > Incorrect Dispensing; (4) We'll file an investigation with the ATM operator. They review the physical cash count and camera footage. Resolution: 5-15 business days. Your claim is stronger if reported within 24 hours.",
        },
        {
            "q": "I accidentally sent {amount} via Nequi but typed the wrong phone number. The money went to a stranger.",
            "a": "Mistaken Nequi transfer process: (1) Contact us immediately — time is critical; (2) We'll send a 'courtesy recall' notification to the recipient asking them to return the funds voluntarily; (3) If they don't respond within 48 hours, we escalate to a formal recall request; (4) IMPORTANT: We cannot forcibly reverse Nequi transfers once completed — the recipient must consent; (5) For amounts over $500,000, consider filing a police report (denuncia) which strengthens recovery. Prevention tip: Always double-check the recipient name that appears before confirming.",
        },
        {
            "q": "I'm being charged {amount} monthly for something called '{merchant}' but I never signed up for this.",
            "a": "Unrecognized recurring charges are common. Steps: (1) Check if you signed up for a free trial that converted to paid — companies like {merchant} often do this after 7-30 day trials; (2) Check your email for any sign-up confirmations from {merchant}; (3) Search '{merchant} cancel subscription' online to find their cancellation page; (4) If you truly never signed up: block the merchant in the app (Transactions > Merchant > Block future charges) and dispute the last 2-3 months of charges; (5) We can recover up to 120 days of unauthorized recurring charges.",
        },
    ]

    template = rng.choice(qa_templates)
    category = rng.choice(list(MERCHANTS_BY_CATEGORY.keys()))
    merchant_name = rng.choice(MERCHANTS_BY_CATEGORY[category])[0]
    amount = _format_amount(rng.randint(10_000, 500_000) // 1000 * 1000)
    recipients = ["María García", "Carlos López", "un contacto", "Juan Rodríguez", "Andrea Mejía"]
    recipient = rng.choice(recipients)

    q = template["q"].format(amount=amount, merchant=merchant_name, recipient=recipient)
    a = template["a"].format(amount=amount, merchant=merchant_name, recipient=recipient)

    return {
        "instruction": "Answer the following customer question about their bank transactions.",
        "input": q,
        "output": a,
    }


def _generate_transaction_explanation(rng: random.Random) -> dict:
    """Generate transaction status explanation examples."""
    scenarios = [
        {
            "status": "PENDING",
            "type": "authorization hold",
            "merchants": ["HOTEL BOOKING.COM", "HERTZ RENT A CAR", "SHELL GASOLINA", "RESTAURANTE EL CIELO", "PARKING ZONA AZUL"],
            "explanation": "This is an authorization hold (pre-authorization) — {amount} is temporarily reserved but NOT deducted from your available balance. {merchant} placed this hold to guarantee the final charge.",
            "resolution": "The hold will either convert to a final charge (usually within 1-3 days) or auto-release if not captured (7-14 business days for hotels, 3-5 days for gas stations).",
        },
        {
            "status": "PENDING",
            "type": "refund processing",
            "merchants": ["MERCADOLIBRE", "FALABELLA.COM.CO", "AMAZON.COM", "ALKOSTO", "ZARA COLOMBIA"],
            "explanation": "This is a refund (devolución) of {amount} from {merchant} being credited back to your account. The refund has been initiated by the merchant.",
            "resolution": "Refunds typically take 5-10 business days to fully process and appear in your available balance. Credit card refunds may take up to 2 billing cycles.",
        },
        {
            "status": "PROCESSING",
            "type": "international settlement",
            "merchants": ["AMAZON.COM USA", "STEAM PURCHASE", "APPLE.COM/BILL", "PAYPAL *INTL", "ALIEXPRESS"],
            "explanation": "This international transaction of {amount} from {merchant} is in settlement processing. International charges go through currency conversion and cross-border clearing.",
            "resolution": "International settlements take 2-5 business days. The final COP amount may differ slightly from the pending amount due to exchange rate fluctuation between authorization and settlement.",
        },
        {
            "status": "DECLINED → PENDING",
            "type": "retry after soft decline",
            "merchants": ["NETFLIX.COM", "SPOTIFY", "YOUTUBE PREMIUM", "DISNEY+", "HBO MAX"],
            "explanation": "Your subscription payment to {merchant} for {amount} was initially declined (soft decline due to temporary processing issue) and automatically retried successfully.",
            "resolution": "No action needed — the payment went through on retry. If you see a duplicate pending charge, the failed attempt will auto-release within 48 hours.",
        },
        {
            "status": "PARTIAL",
            "type": "partial authorization",
            "merchants": ["EXITO", "CARULLA", "JUMBO", "OLIMPICA", "MAKRO"],
            "explanation": "The purchase at {merchant} was partially authorized: your card covered {amount} of the total. The remaining balance was either paid with another method or the purchase was adjusted.",
            "resolution": "Check your receipt — some supermarkets allow split payments. If you expected the full amount to be charged, verify your daily transaction limit hasn't been reached.",
        },
    ]

    scenario = rng.choice(scenarios)
    merchant = rng.choice(scenario["merchants"])
    amount = _format_amount(rng.randint(20_000, 1_500_000) // 1000 * 1000)
    days_pending = rng.randint(1, 7)

    return {
        "instruction": "Explain what the following pending transaction means and when it will be finalized.",
        "input": f"{scenario['status']}: {merchant} - {amount} - Status: {scenario['type'].title()} since {days_pending} day{'s' if days_pending > 1 else ''}.",
        "output": f"{scenario['explanation'].format(amount=amount, merchant=merchant)} {scenario['resolution']}",
    }


def _generate_merchant_identification(rng: random.Random) -> dict:
    """Generate unknown merchant identification examples."""
    merchant_mappings = [
        ("PG*IFOOD COL", "iFood Colombia (food delivery app)", "'PG*' = payment gateway prefix. iFood is a Brazilian food delivery platform operating in Colombia."),
        ("GOOGLE *SERVICES", "Google paid service (Google One storage, YouTube Premium, Play subscription, or Google Workspace)", "Check pay.google.com > Subscriptions to see which specific Google service is charging you."),
        ("MSFT *MICROSOFT", "Microsoft subscription (Microsoft 365, Xbox Game Pass, OneDrive, or Azure)", "Check account.microsoft.com > Services & subscriptions for details."),
        ("AMZN MKTP CO", "Amazon Marketplace Colombia purchase", "'MKTP' = Marketplace (third-party seller on Amazon). Check your Amazon order history for the matching amount."),
        ("SQ *TIENDA", "Square payment terminal (small business)", "'SQ*' = processed through Square POS. Common for small shops, food trucks, and independent vendors."),
        ("PAY*MERCADOPAGO", "MercadoPago wallet payment (tied to MercadoLibre)", "'PAY*' = MercadoPago payment processor. Check your MercadoPago app activity for the specific purchase."),
        ("CL*RAPPI TURBO", "Rappi Turbo (express 10-min delivery)", "'CL*' = clearing house prefix. Rappi Turbo is the ultra-fast delivery option within Rappi."),
        ("DLOCAL*UBER", "Uber ride or Uber Eats order", "'DLOCAL*' = dLocal payment processor (handles Uber's payments in Latin America). Check your Uber app trip/order history."),
        ("PP*CANVA", "Canva design platform subscription", "'PP*' = processed via PayPal. Check if you signed up for Canva Pro or Canva for Teams."),
        ("WPLAY.CO", "Wplay online betting platform (sports/casino)", "Wplay is a licensed Colombian online gambling platform. If unrecognized, check if household members have an account."),
        ("CREDIBANC*POS", "Point-of-sale purchase at a small merchant using CredibanCo terminal", "CredibanCo is a Colombian payment processor. The specific merchant depends on the date/amount — check your location history."),
        ("NEQUI*CASHIN", "Nequi cash-in transaction (deposited cash at a physical point)", "This is money you loaded into your Nequi wallet at a Bancolombia ATM or authorized cash-in point."),
        ("PSE*ACH DEBIT", "Automated Clearing House debit via PSE", "This is a direct debit authorized for a recurring bill or subscription payment through Colombia's PSE system."),
        ("APPSTORE PURCHASE", "Apple App Store purchase (app, in-app purchase, or subscription)", "Check Settings > Apple ID > Subscriptions on your iPhone, or reportaproblem.apple.com for purchase history."),
    ]

    mapping = rng.choice(merchant_mappings)
    merchant_code, identity, explanation = mapping
    amount = _format_amount(rng.randint(5_000, 200_000) // 1000 * 1000)
    date = _random_date(rng)

    return {
        "instruction": "The customer doesn't recognize this merchant name on their statement. Help identify it.",
        "input": f"Unknown charge: '{merchant_code}' - {amount} - {date}",
        "output": f"'{merchant_code}' is {identity}. {explanation} Amount {amount} on {date.split()[0]}. If still unrecognized after checking, dispute within 120 days.",
    }


def _generate_financial_planning(rng: random.Random) -> dict:
    """Generate savings/budget recommendation examples."""
    profile = rng.choice(CUSTOMER_PROFILES)
    income = profile["avg_monthly"] + rng.randint(500_000, 2_000_000)
    expenses = int(income * rng.uniform(0.7, 0.95))
    surplus = income - expenses

    goals = [
        ("emergency fund", rng.randint(10_000_000, 20_000_000), "3-6 months of expenses"),
        ("vacation savings", rng.randint(3_000_000, 8_000_000), "travel fund"),
        ("new laptop", rng.randint(3_000_000, 6_000_000), "tech purchase"),
        ("apartment deposit", rng.randint(5_000_000, 15_000_000), "housing"),
        ("investment portfolio", rng.randint(5_000_000, 20_000_000), "long-term wealth building"),
        ("debt payoff", rng.randint(2_000_000, 10_000_000), "credit card/loan balance"),
    ]
    goal_name, goal_amount, goal_type = rng.choice(goals)

    # Identify top discretionary categories
    discretionary = rng.sample(
        ["food delivery (12x/month)", "subscriptions (5 active)", "nightlife (3x/month)",
         "online shopping", "gaming/entertainment", "café visits (daily)"],
        k=rng.randint(2, 3)
    )
    discretionary_text = ", ".join(discretionary)

    months_current = round(goal_amount / max(surplus, 100_000), 1)
    optimized_savings = int(surplus * 1.6)
    months_optimized = round(goal_amount / max(optimized_savings, 100_000), 1)

    return {
        "instruction": "Based on the customer's transaction history, suggest a personalized savings plan.",
        "input": f"Profile: Income {_format_amount(income)}/month, expenses average {_format_amount(expenses)}. Current savings: {_format_amount(0)}. Recurring discretionary: {discretionary_text}. Goal: {goal_name} {_format_amount(goal_amount)} ({goal_type}).",
        "output": f"Current surplus: {_format_amount(surplus)}/month ({months_current} months to goal at current rate). Optimizations: reduce top 2 discretionary categories by 40% to free up ~{_format_amount(optimized_savings - surplus)}/month. New capacity: {_format_amount(optimized_savings)}/month. Revised timeline: {months_optimized} months. Set up automatic {_format_amount(optimized_savings)} transfer on payday to a dedicated savings pocket.",
    }


def _generate_dispute_resolution(rng: random.Random) -> dict:
    """Generate chargeback/dispute guidance examples."""
    dispute_scenarios = [
        {
            "situation": "bought {item} online from {merchant} for {amount} but never received it. It's been {days} days",
            "dispute_type": "merchandise not received",
            "steps": "(1) Gather order confirmation, tracking info (if any), and screenshots of merchant contact attempts; (2) You're within the 120-day dispute window; (3) File '{dispute_type}' chargeback: Transactions > Select charge > Report Problem > Item not received; (4) Provisional credit within 5 business days while we investigate (30-60 days).",
            "items": ["shoes", "a phone case", "headphones", "a backpack", "a jacket", "electronics"],
        },
        {
            "situation": "was charged {amount} by {merchant} but the item arrived damaged/defective and they refuse to refund",
            "dispute_type": "defective merchandise",
            "steps": "(1) Document the damage with photos/video; (2) Save all communication with {merchant} showing they refused the refund; (3) File '{dispute_type}' dispute: Transactions > Report Problem > Item defective; (4) We'll request evidence from the merchant — they have 30 days to respond; (5) If they don't respond or can't prove delivery of undamaged goods, you win the dispute.",
            "items": ["a blender", "clothing", "a chair", "a monitor", "shoes", "a speaker"],
        },
        {
            "situation": "was charged {amount} twice by {merchant} for the same purchase — one is a duplicate",
            "dispute_type": "duplicate charge",
            "steps": "(1) Verify both charges posted (not one pending + one posted, which is normal); (2) Check your email for two separate order confirmations vs one; (3) File '{dispute_type}' dispute for the extra charge; (4) Duplicate charges are straightforward — resolution typically within 10 business days; (5) Keep your single receipt as evidence.",
            "items": ["a meal", "groceries", "a ride", "a subscription", "a purchase", "gas"],
        },
        {
            "situation": "cancelled a subscription with {merchant} but was charged {amount} after cancellation",
            "dispute_type": "charge after cancellation",
            "steps": "(1) Get proof of cancellation (email, screenshot of cancellation confirmation, date); (2) Check if the charge covers a period before cancellation (some services bill in advance); (3) If charge is genuinely post-cancellation: File '{dispute_type}' dispute; (4) Also block future charges from this merchant: Transactions > Merchant > Block; (5) Resolution: 15-30 business days. We can recover up to 120 days of post-cancellation charges.",
            "items": [],
        },
        {
            "situation": "was charged {amount} by {merchant} but the actual amount should have been {lesser_amount} — they overcharged",
            "dispute_type": "incorrect amount",
            "steps": "(1) Keep your receipt showing the correct amount of {lesser_amount}; (2) Contact {merchant} first — many will correct overcharges directly within 5 days; (3) If merchant doesn't resolve: File '{dispute_type}' dispute with receipt as evidence; (4) We'll dispute only the difference ({amount} - {lesser_amount}); (5) Partial disputes resolve in 15-30 business days.",
            "items": [],
        },
    ]

    scenario = rng.choice(dispute_scenarios)
    category = rng.choice(list(MERCHANTS_BY_CATEGORY.keys()))
    merchant = rng.choice(MERCHANTS_BY_CATEGORY[category])[0]
    amount = _format_amount(rng.randint(30_000, 800_000) // 1000 * 1000)
    days = rng.randint(7, 45)
    item = rng.choice(scenario["items"]) if scenario["items"] else ""
    lesser_amount = _format_amount(rng.randint(10_000, 300_000) // 1000 * 1000)

    situation = scenario["situation"].format(
        item=item, merchant=merchant, amount=amount, days=days, lesser_amount=lesser_amount
    )
    steps = scenario["steps"].format(
        merchant=merchant, dispute_type=scenario["dispute_type"], lesser_amount=lesser_amount, amount=amount
    )

    return {
        "instruction": "Guide the customer through disputing the following transaction.",
        "input": f"Customer says: I {situation}.",
        "output": f"Dispute type: {scenario['dispute_type'].upper()}. {steps}",
    }


def _generate_payment_failure(rng: random.Random) -> dict:
    """Generate declined payment explanation examples."""
    decline_reasons = [
        {
            "code": "INSUFFICIENT_FUNDS",
            "explanation": "Your available balance ({balance}) is less than the purchase amount ({amount}). Note: pending holds reduce available balance even if your total balance looks sufficient.",
            "fix": "Transfer funds from a savings pocket or another account, then retry. Pro tip: enable 'Auto-backup from pocket' to automatically cover shortfalls.",
        },
        {
            "code": "INTERNATIONAL_BLOCK",
            "explanation": "International purchases are blocked by your security settings. {merchant} processes payments from abroad even though it may appear local.",
            "fix": "App > Cards > International purchases > Toggle ON > Retry immediately > Toggle OFF after for security. Takes effect instantly.",
        },
        {
            "code": "DAILY_LIMIT_EXCEEDED",
            "explanation": "You've reached your daily spending limit. Previous transactions today total {daily_spent}, and adding {amount} exceeds your {limit} daily cap.",
            "fix": "Options: (1) Wait until midnight when limits reset; (2) Request a temporary limit increase: Settings > Limits > Temporary increase (requires biometric); (3) Use an alternative payment method for now.",
        },
        {
            "code": "CARD_EXPIRED",
            "explanation": "Your physical/virtual card has expired. Expired cards are automatically blocked from new transactions.",
            "fix": "Request a replacement card: Cards > Manage > Request new card. Virtual cards are issued instantly; physical cards arrive in 5-7 business days. Update any saved card details on merchant sites.",
        },
        {
            "code": "SUSPECTED_FRAUD",
            "explanation": "Our fraud system flagged this transaction as potentially unauthorized based on unusual patterns (location, amount, time, or merchant type).",
            "fix": "If this was you: (1) Check for an in-app verification prompt — approve it to whitelist the transaction; (2) Retry immediately after approving; (3) If no prompt appears, call support to unlock. Future similar transactions will be whitelisted.",
        },
        {
            "code": "MERCHANT_CATEGORY_BLOCKED",
            "explanation": "You have category restrictions active that block {merchant_category} purchases. This is either a parental control or a self-imposed spending control.",
            "fix": "Check Settings > Cards > Spending Controls to see which categories are blocked. You can temporarily enable the category, make your purchase, then re-block it.",
        },
        {
            "code": "ONLINE_PURCHASES_DISABLED",
            "explanation": "Online/e-commerce purchases are currently disabled for your card. This security setting blocks all card-not-present transactions.",
            "fix": "App > Cards > Online purchases > Toggle ON. Make your purchase, then toggle OFF for security. Some users keep this off by default and only enable per-transaction.",
        },
    ]

    reason = rng.choice(decline_reasons)
    category = rng.choice(list(MERCHANTS_BY_CATEGORY.keys()))
    merchant = rng.choice(MERCHANTS_BY_CATEGORY[category])[0]
    amount = _format_amount(rng.randint(20_000, 2_000_000) // 1000 * 1000)
    balance = _format_amount(rng.randint(10_000, 500_000) // 1000 * 1000)
    daily_spent = _format_amount(rng.randint(1_000_000, 4_000_000) // 1000 * 1000)
    limit = _format_amount(rng.randint(3_000_000, 5_000_000) // 1000 * 1000)

    explanation = reason["explanation"].format(
        amount=amount, merchant=merchant, balance=balance,
        daily_spent=daily_spent, limit=limit, merchant_category=category
    )

    return {
        "instruction": "Explain why this payment was declined and suggest next steps.",
        "input": f"Declined: {merchant} - {amount} - Reason: {reason['code']}. Balance: {balance}.",
        "output": f"Decline reason: {reason['code']}. {explanation} Fix: {reason['fix']}",
    }


def _generate_account_security(rng: random.Random) -> dict:
    """Generate account security assessment examples."""
    security_events = [
        {
            "events": "{n_attempts} failed login attempts from {city} IP (customer lives in {home_city})",
            "additional": "password not changed in {months} months, 2FA is {twofa_status}",
            "risk": "MODERATE",
            "analysis": "Someone is testing credentials against your account. The failed attempts from {city} suggest a targeted or credential-stuffing attack.",
            "actions": "(1) Change password NOW (use 12+ characters with mixed case, numbers, symbols); (2) Enable 2FA immediately (Settings > Security > Two-factor); (3) Check active sessions and remove unknown devices; (4) Set up login alerts; (5) Check if your email appears in known breaches at haveibeenpwned.com.",
        },
        {
            "events": "New device logged in from {city}, password changed, and email updated within {minutes} minutes",
            "additional": "Previous session terminated, recovery email changed to unknown address",
            "risk": "CRITICAL",
            "analysis": "Account takeover in progress. The attacker has gained access, changed credentials, and is locking you out by changing the recovery email.",
            "actions": "(1) CALL support immediately at the emergency line — do NOT use in-app chat (attacker may intercept); (2) We'll freeze the account to prevent fund movement; (3) Visit a physical branch with your ID for identity verification; (4) All recent transactions will be reviewed; (5) A new account recovery process will be initiated with verified contact info.",
        },
        {
            "events": "Login from {city} on new device at {time}, followed by beneficiary list export and {n_transfers} new beneficiaries added",
            "additional": "All activity within last {minutes} minutes, customer hasn't logged in for {days} days",
            "risk": "HIGH",
            "analysis": "Potential account compromise preparing for fund extraction. Adding multiple beneficiaries after a dormant period is a classic pre-fraud pattern.",
            "actions": "(1) Block all outgoing transfers immediately; (2) Remove all recently added beneficiaries; (3) Force session termination on all devices; (4) Require in-person or video-call identity verification before re-enabling transfers; (5) Review the past {days} days for any unauthorized activity.",
        },
        {
            "events": "3 successful logins from {city}, {city2}, and {city3} within same hour",
            "additional": "All on different devices, same browser fingerprint pattern",
            "risk": "HIGH",
            "analysis": "Geographic impossibility — simultaneous sessions from 3 different cities indicate credential sharing or compromise. The same browser fingerprint suggests automated tools.",
            "actions": "(1) Terminate ALL active sessions immediately; (2) Force password reset; (3) Enable mandatory 2FA; (4) Review all transactions from the past 24 hours; (5) If any unauthorized activity found, file a fraud report and reverse transactions.",
        },
        {
            "events": "Customer reports receiving OTP codes they didn't request ({n_attempts} in last hour)",
            "additional": "No login attempts visible from customer's devices, SIM status unchanged",
            "risk": "MODERATE",
            "analysis": "Someone has your phone number and is attempting to trigger OTP-based actions (login, password reset, or transaction approval). They do NOT yet have access.",
            "actions": "(1) Do NOT share the OTP codes with anyone (social engineering risk); (2) Change your password preemptively; (3) Verify your SIM hasn't been ported (contact carrier); (4) Enable app-based 2FA instead of SMS-based; (5) Consider changing your linked phone number if this persists.",
        },
    ]

    event = rng.choice(security_events)
    cities = ["Medellín", "Cali", "Barranquilla", "Cartagena", "Bucaramanga", "Pereira", "Santa Marta"]
    home_cities = ["Bogotá", "Medellín", "Cali"]
    home_city = rng.choice(home_cities)
    city = rng.choice([c for c in cities if c != home_city])
    city2 = rng.choice([c for c in cities if c not in (home_city, city)])
    city3 = rng.choice([c for c in cities if c not in (home_city, city, city2)])
    n_attempts = rng.randint(3, 15)
    months = rng.randint(6, 24)
    minutes = rng.randint(5, 30)
    days = rng.randint(7, 90)
    time = f"{rng.randint(0, 5):02d}:{rng.randint(0, 59):02d} AM"
    twofa_status = rng.choice(["OFF", "SMS-only (weak)"])

    events_text = event["events"].format(
        n_attempts=n_attempts, city=city, home_city=home_city,
        minutes=minutes, time=time, city2=city2, city3=city3,
        n_transfers=rng.randint(3, 8), days=days
    )
    additional_text = event["additional"].format(
        months=months, twofa_status=twofa_status, minutes=minutes, days=days
    )
    analysis = event["analysis"].format(city=city, days=days)
    actions = event["actions"].format(days=days)

    return {
        "instruction": "Review these recent account security events and advise the customer on their account safety.",
        "input": f"Security log: {events_text}. Additional context: {additional_text}.",
        "output": f"{event['risk']} RISK. {analysis} Immediate actions: {actions}",
    }


def _generate_additional_examples(num_needed: int, seed: int = 42) -> list[dict]:
    """Generate unique parameterized examples to supplement handcrafted ones."""
    rng = random.Random(seed)
    generated = []
    seen = set()

    generators = [
        (_generate_categorization, 0.18),
        (_generate_fraud_analysis, 0.14),
        (_generate_spending_insight, 0.09),
        (_generate_customer_qa, 0.14),
        (_generate_transaction_explanation, 0.10),
        (_generate_merchant_identification, 0.10),
        (_generate_financial_planning, 0.08),
        (_generate_dispute_resolution, 0.07),
        (_generate_payment_failure, 0.05),
        (_generate_account_security, 0.05),
    ]

    attempts = 0
    max_attempts = num_needed * 5

    while len(generated) < num_needed and attempts < max_attempts:
        attempts += 1
        # Weighted random selection of generator
        r = rng.random()
        cumulative = 0
        for gen_fn, weight in generators:
            cumulative += weight
            if r <= cumulative:
                example = gen_fn(rng)
                break

        # Deduplicate by input
        key = example["input"]
        if key not in seen:
            seen.add(key)
            generated.append(example)

    return generated


def create_sample_data(output_path: Path, num_samples: int = 1000) -> None:
    """Create sample training data for demonstration."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Handcrafted high-quality examples (87 templates) ---
    examples = [
        # --- Transaction Categorization ---
        {
            "instruction": "Categorize the following bank transaction based on its description.",
            "input": "RAPPI*RAPPI COL - $45,000 COP - 2024-03-15 20:32",
            "output": "Category: Food & Delivery. This is a food delivery purchase through the Rappi platform.",
        },
        {
            "instruction": "Categorize the following bank transaction based on its description.",
            "input": "NETFLIX.COM - $38,900 COP - 2024-03-01 00:01",
            "output": "Category: Entertainment & Subscriptions. This is a recurring monthly subscription to Netflix streaming service.",
        },
        {
            "instruction": "Categorize the following bank transaction based on its description.",
            "input": "EXITO CALLE 80 - $187,500 COP - 2024-03-12 14:15",
            "output": "Category: Groceries & Supermarket. This is a purchase at Éxito supermarket, a major retail chain.",
        },
        {
            "instruction": "Categorize the following bank transaction based on its description.",
            "input": "UBER *TRIP - $12,300 COP - 2024-03-14 07:45",
            "output": "Category: Transportation. This is a ride-hailing trip payment through Uber.",
        },
        {
            "instruction": "Categorize the following bank transaction based on its description.",
            "input": "TRANSFERENCIA A JUAN PEREZ - $500,000 COP - 2024-03-10 09:00",
            "output": "Category: Transfers & Payments. This is a person-to-person transfer to another individual.",
        },
        {
            "instruction": "Categorize the following bank transaction based on its description.",
            "input": "PSE ENEL CODENSA - $145,000 COP - 2024-03-05 10:00",
            "output": "Category: Utilities & Bills. This is an electricity bill payment to Enel-Codensa via PSE (online banking transfer).",
        },
        # --- Fraud Detection ---
        {
            "instruction": "Analyze this transaction and determine if it shows signs of potential fraud. Explain your reasoning.",
            "input": "Transaction: POS PURCHASE ELECTRONICS STORE - $3,200,000 COP - 2024-03-15 03:47 AM. Customer profile: Average monthly spending $800,000 COP, no previous electronics purchases above $500,000.",
            "output": "HIGH RISK - Potential fraud indicators: (1) Unusual time — purchase at 3:47 AM is outside normal activity hours; (2) Amount significantly exceeds typical spending pattern (4x monthly average in single transaction); (3) No prior history of high-value electronics purchases. Recommendation: Flag for verification and send real-time alert to customer.",
        },
        {
            "instruction": "Analyze this transaction and determine if it shows signs of potential fraud. Explain your reasoning.",
            "input": "Transaction: SUPERMERCADO CARULLA - $95,000 COP - 2024-03-15 18:30. Customer profile: Regular weekly grocery shopper, average grocery spend $80,000-$120,000 per visit.",
            "output": "LOW RISK - This transaction appears legitimate: (1) Known merchant the customer frequents; (2) Amount is within normal spending range; (3) Time of day is consistent with typical shopping hours; (4) Category matches established behavior pattern. No action needed.",
        },
        {
            "instruction": "Analyze this transaction and determine if it shows signs of potential fraud. Explain your reasoning.",
            "input": "Transaction: ATM WITHDRAWAL CARTAGENA - $600,000 COP - 2024-03-15 22:10. Customer profile: Lives in Bogotá, no recent travel bookings, last ATM use was in Bogotá 4 hours ago.",
            "output": "HIGH RISK - Potential fraud indicators: (1) Geographic impossibility — ATM withdrawal in Cartagena while last activity was in Bogotá 4 hours prior; (2) Evening withdrawal at distant location; (3) Amount above average ATM withdrawal. Recommendation: Block card immediately and contact customer.",
        },
        {
            "instruction": "Analyze this transaction and determine if it shows signs of potential fraud. Explain your reasoning.",
            "input": "Transaction: 15 transfers of exactly $999,000 COP each within 2 hours to different accounts. Customer profile: Received $15,000,000 from unknown source yesterday.",
            "output": "CRITICAL - Structuring/smurfing pattern detected: (1) Multiple transfers just below the $1,000,000 reporting threshold; (2) Rapid succession to different recipients; (3) Large unexplained incoming deposit. Immediate action: Block all outgoing transfers, report to UIAF compliance team.",
        },
        {
            "instruction": "Analyze this transaction and determine if it shows signs of potential fraud. Explain your reasoning.",
            "input": "Transaction: SIM SWAP detected — new device logged into account, followed by password change and transfer of $3,000,000 to new beneficiary within 8 minutes.",
            "output": "CRITICAL - SIM swap fraud in progress: (1) Device change + password change + large transfer in 8 minutes is textbook SIM swap attack; (2) New beneficiary added and funded immediately; (3) Speed indicates automated/scripted attack. Immediate action: Block account completely, reverse pending transfer, contact customer via alternate channel.",
        },
        # --- Customer Support Q&A ---
        {
            "instruction": "Answer the following customer question about their bank transactions.",
            "input": "I see a charge of $38,900 from NETFLIX.COM but I cancelled my subscription last month. What should I do?",
            "output": "This appears to be a charge after cancellation. Here's what to do: (1) Verify the cancellation date in your Netflix account settings — subscriptions typically remain active until the current billing cycle ends; (2) If the charge is after your cycle ended, dispute through the app under 'Transaction Details > Report a Problem'; (3) We'll investigate within 5 business days and issue a provisional credit.",
        },
        {
            "instruction": "Answer the following customer question about their bank transactions.",
            "input": "What does PSE mean in my transaction history? I see PSE ETB and PSE ENEL.",
            "output": "PSE stands for 'Pagos Seguros en Línea' (Secure Online Payments). It's Colombia's electronic payment system for paying bills directly from your bank account online. 'PSE ETB' = phone/internet bill, 'PSE ENEL' = electricity. These are legitimate online bill payments.",
        },
        {
            "instruction": "Answer the following customer question about their bank transactions.",
            "input": "I received a text saying my account is blocked and to click a link to verify. Is this real?",
            "output": "This is almost certainly a PHISHING attempt. DO NOT click the link. We NEVER send links via SMS asking you to verify your account. If your account were truly blocked, you'd see it in the app. Delete the message, open your app normally to confirm your account is active, and report it in-app (Help > Report Fraud > Phishing).",
        },
        {
            "instruction": "Answer the following customer question about their bank transactions.",
            "input": "How can I increase my daily transfer limit? I need to send $5,000,000 but my limit is $3,000,000.",
            "output": "Options: (1) TEMPORARY INCREASE: Settings > Limits > Request temporary increase — up to $10,000,000 for 24 hours with biometric verification; (2) PERMANENT INCREASE: Requires income verification (2 pay stubs), approved within 48 hours; (3) SPLIT: Send $3,000,000 today, $2,000,000 tomorrow (limits reset at midnight).",
        },
        {
            "instruction": "Answer the following customer question about their bank transactions.",
            "input": "I made a transfer of $300,000 to the wrong account number. Can I get my money back?",
            "output": "We'll file a 'recall request' with the receiving bank: (1) If funds are still there, recovery takes 5-15 business days; (2) If already withdrawn, it may require legal action; (3) For amounts over $200,000, also file a report with the SFC (Superintendencia Financiera). Time is critical — the sooner we act, the better the chances.",
        },
        # --- Spending Insights ---
        {
            "instruction": "Analyze the following monthly transaction summary and provide spending insights and recommendations.",
            "input": "Monthly Summary (March 2024): Food & Delivery: $620,000 (28%), Transportation: $280,000 (13%), Subscriptions: $95,000 (4%), Groceries: $450,000 (20%), Utilities: $320,000 (15%). Total: $2,195,000 COP.",
            "output": "Key Insights: (1) Combined food spending (Food + Groceries = 48%) is nearly half your budget — main optimization area; (2) Subscriptions well-controlled at 4%; (3) Transportation reasonable. Recommendation: Set a weekly food delivery budget of $100,000 and meal prep 2-3 days/week to reduce delivery spending by ~$200,000/month.",
        },
        # --- Transaction Explanation ---
        {
            "instruction": "Explain what the following pending transaction means and when it will be finalized.",
            "input": "PENDING: AUTH HOLD - HOTEL BOOKING.COM - $850,000 COP - Status: Pending since 3 days",
            "output": "This is an authorization hold (pre-authorization), not a final charge. The $850,000 is temporarily reserved but not deducted. Hotel holds finalize at checkout with the actual amount. If pending for 3 days, it should finalize or auto-release within 7-14 business days.",
        },
        {
            "instruction": "Explain what the following pending transaction means and when it will be finalized.",
            "input": "PENDING: DEVOLUCION - MERCADOLIBRE - $125,000 COP - Status: Processing",
            "output": "This is a refund (devolución) from MercadoLibre — $125,000 will be credited back. Refunds typically take 5-10 business days to fully process. Once status changes to 'Completed', funds are available in your balance.",
        },
        # --- Dispute Resolution ---
        {
            "instruction": "Guide the customer through disputing the following transaction.",
            "input": "Customer says: I bought shoes online from TIENDA XYZ for $180,000 but never received them. It's been 3 weeks.",
            "output": "Dispute steps: (1) Gather order confirmation, tracking info, screenshots of merchant contact attempts; (2) You're within the 120-day dispute window; (3) File 'merchandise not received' chargeback: Transactions > Select charge > Report Problem; (4) Provisional credit within 5 business days, investigation takes 30-60 days.",
        },
        # --- International Transactions ---
        {
            "instruction": "Explain the fees and exchange rate applied to this international transaction.",
            "input": "Transaction: AMAZON.COM USA - Original: $45.99 USD - Charged: $198,500 COP - Exchange rate: $4,316.15 COP/USD - International fee: $5,950 COP.",
            "output": "Breakdown: $45.99 USD × $4,316.15 rate = ~$198,500 COP + $5,950 international fee (3%) = $204,450 total. Tips: (1) Always choose to be charged in USD at checkout (merchant rates are worse); (2) Consider a card with no foreign transaction fee for recurring international purchases.",
        },
        # --- Financial Planning ---
        {
            "instruction": "Based on the customer's transaction history, suggest a personalized savings plan.",
            "input": "Profile: Income $5,200,000/month, expenses average $4,800,000. Current savings: $0. Recurring discretionary: Rappi 12x/month ($540,000), streaming ($120,000), nightlife 3x/month ($300,000). Goal: Emergency fund $15,000,000.",
            "output": "Current surplus: $400,000/month (37.5 months to goal). Optimizations: (1) Rappi 12x→6x saves $270,000; (2) Cut 1-2 streaming services saves $50,000; (3) Nightlife 3→2 outings saves $100,000. New capacity: $820,000/month. Revised timeline: 18.3 months. Set up automatic $820,000 transfer on payday.",
        },
        # --- Account Security ---
        {
            "instruction": "Review these recent account security events and advise the customer on their account safety.",
            "input": "Security log: 3 failed login attempts from Medellín IP (customer lives in Bogotá), password not changed in 14 months, 2FA is OFF.",
            "output": "MODERATE RISK. Someone is trying your credentials. Immediate actions: (1) Change password NOW (12+ characters); (2) Enable 2FA (Settings > Security); (3) Check active sessions and remove unknown devices; (4) Set login alerts. The failed attempts suggest a possible credential leak — check haveibeenpwned.com.",
        },
        # --- Payment Failures ---
        {
            "instruction": "Explain why this payment was declined and suggest next steps.",
            "input": "Declined: BOOKING.COM HOTEL - $1,200,000 COP - Reason: INTERNATIONAL_BLOCK. Balance: $3,500,000 (sufficient).",
            "output": "Blocked by international transaction security (not a balance issue). Booking.com processes through Netherlands. Fix: App > Cards > International purchases > Toggle ON > Retry immediately > Toggle OFF after. Takes 30 seconds.",
        },
        {
            "instruction": "Explain why this payment was declined and suggest next steps.",
            "input": "Declined: AMAZON.COM - $350,000 COP - Reason: INSUFFICIENT_FUNDS. Balance: $280,000. Savings pocket: $1,200,000.",
            "output": "Balance ($280,000) < purchase ($350,000). Savings pockets don't auto-cover purchases (by design). Fix: Transfer $70,000+ from pocket to main balance (Settings > Pockets > Transfer), then retry. Pro tip: Enable 'Auto-backup from pocket' to avoid future declines.",
        },
        # --- Merchant Identification ---
        {
            "instruction": "The customer doesn't recognize this merchant name on their statement. Help identify it.",
            "input": "Unknown charge: 'PG*IFOOD COL' - $38,500 COP - 2024-03-15 13:22",
            "output": "'PG*IFOOD COL' is iFood Colombia (food delivery). 'PG*' = payment gateway prefix. Verify: check if you have iFood installed, the 1:22 PM timestamp matches lunch, $38,500 is typical for single-meal delivery. If unrecognized, check household members or dispute.",
        },
        {
            "instruction": "The customer doesn't recognize this merchant name on their statement. Help identify it.",
            "input": "Unknown charge: 'GOOGLE *SERVICES' - $10,900 COP - 2024-03-01 00:04",
            "output": "'GOOGLE *SERVICES' = Google paid service (Google One storage, YouTube Premium, or Play subscription). $10,900 matches Google One 100GB in Colombia. Verify at pay.google.com > Subscriptions. Common accidental subscription when Google warns your Drive is full.",
        },
        # --- Tax Analysis ---
        {
            "instruction": "Identify tax-deductible transactions from this customer's annual history and estimate potential tax savings.",
            "input": "2023 flagged: Health (EPS): $3,600,000; Education (ICETEX): $5,040,000; Mortgage interest: $15,840,000; Charity: $1,200,000; Voluntary pension: $6,000,000.",
            "output": "Estimated deductions: $31,680,000. At 28% marginal rate: ~$8,870,000 tax savings. All categories are valid under Colombian tax code (ET). Ensure DIAN-certified receipts are collected before filing deadline. Export categorized report: Reports > Annual Tax Summary.",
        },
        # --- Credit Analysis ---
        {
            "instruction": "Analyze this customer's credit card usage pattern and assess their credit health.",
            "input": "Credit card: Limit $5,000,000. Balance: $4,200,000. Minimum payment: $168,000/month. Interest: 2.1%/month. Usage: 60% food delivery, 25% subscriptions, 15% cash advances.",
            "output": "CRITICAL: 84% utilization (should be <30%), paying minimums only = 47 months to clear + $3,730,000 in interest. Recovery plan: (1) Stop cash advances (highest interest); (2) Pay $400,000/month → debt-free in 13 months; (3) Move food delivery to debit; (4) Target 30% utilization ($1,500,000 balance) within 6 months for credit score recovery.",
        },
    ]

    # Generate additional unique examples to reach target
    num_needed = max(0, num_samples - len(examples))
    if num_needed > 0:
        additional = _generate_additional_examples(num_needed, seed=42)
        all_examples = examples + additional
    else:
        all_examples = examples[:num_samples]

    # Shuffle for variety (with fixed seed for reproducibility)
    rng = random.Random(42)
    rng.shuffle(all_examples)

    with open(output_path, "w") as f:
        for ex in all_examples[:num_samples]:
            f.write(json.dumps(ex) + "\n")

    logger.info(f"Created sample dataset: {output_path} ({min(num_samples, len(all_examples))} examples, {len(examples)} handcrafted + {len(all_examples) - len(examples)} generated)")


def split_data(
    input_path: Path,
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict:
    """Split dataset into train/eval/test splits."""
    logger.info(f"Loading data from {input_path}")
    examples = DataLoader.load_dataset(input_path)
    logger.info(f"Loaded {len(examples)} examples")

    # Validate ratios
    total = train_ratio + eval_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    # Split data
    n_train = int(len(examples) * train_ratio)
    n_eval = int(len(examples) * eval_ratio)

    train_examples = examples[:n_train]
    eval_examples = examples[n_train : n_train + n_eval]
    test_examples = examples[n_train + n_eval :]

    # Save splits
    output_dir = input_path.parent.parent / "curated"
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": (output_dir / "train.jsonl", train_examples),
        "eval": (output_dir / "eval.jsonl", eval_examples),
        "test": (output_dir / "test.jsonl", test_examples),
    }

    for split_name, (split_path, split_data) in splits.items():
        with open(split_path, "w") as f:
            for ex in split_data:
                f.write(json.dumps(ex.model_dump()) + "\n")
        logger.info(f"{split_name.upper()}: {split_path} ({len(split_data)} examples)")

    return splits


def validate_and_report(data_dir: Path) -> None:
    """Validate prepared data and print statistics."""
    logger.info("Validating prepared data...")

    for split_file in ["train.jsonl", "eval.jsonl", "test.jsonl"]:
        path = data_dir / "curated" / split_file
        if path.exists():
            stats = DataLoader.validate_dataset(path)
            logger.info(f"\n{split_file}:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.warning(f"File not found: {path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Prepare data for fine-tuning")
    parser.add_argument(
        "--input",
        type=Path,
        default="data/RAW/training_data.jsonl",
        help="Input data file",
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample data if input does not exist",
    )
    parser.add_argument(
        "--num-samples", type=int, default=1000, help="Number of samples to create"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="data",
        help="Root data directory",
    )

    args = parser.parse_args()

    # Create sample data if needed
    if args.create_sample and not args.input.exists():
        logger.info("Creating sample training data...")
        create_sample_data(args.input, num_samples=args.num_samples)

    # Validate input exists
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        logger.error(
            "Use --create-sample to generate sample data, or provide a data file."
        )
        return

    # Split data
    split_data(args.input)

    # Validate and report
    validate_and_report(args.data_dir)
    logger.info("Data preparation complete!")


if __name__ == "__main__":
    main()
