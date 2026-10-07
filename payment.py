"""
Rookie Pass — Razorpay payment routes
======================================
Three endpoints:
  POST /orders                — create a Razorpay order, persist to Supabase
  POST /webhooks/razorpay     — verify signature, call grant_access()
  GET  /orders/{id}/status    — return DB status; self-heals missed webhooks
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────────────
RZP_KEY_ID      = os.environ.get("RAZORPAY_KEY_ID",      "rzp_live_TjpVOv7iwJP6Mj")
RZP_KEY_SECRET  = os.environ.get("RAZORPAY_KEY_SECRET",  "LoI4wu4B0zn6ZvVMxVzx6qS8")
WEBHOOK_SECRET  = os.environ.get("RAZORPAY_WEBHOOK_SECRET", RZP_KEY_SECRET)

SUPABASE_URL    = os.environ.get("SUPABASE_URL", "https://rzcizwacjexolkjjczbt.supabase.co")
SUPABASE_KEY    = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")  # service role — set in env

RZP_BASE        = "https://api.razorpay.com/v1"

router = APIRouter()


# ── Supabase helpers (raw REST — no extra SDK) ─────────────────────────────────

def _sb_headers() -> dict[str, str]:
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }


async def _sb_get(path: str, params: dict | None = None) -> Any:
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{SUPABASE_URL}/rest/v1/{path}",
                        headers=_sb_headers(), params=params or {})
    r.raise_for_status()
    return r.json()


async def _sb_post(path: str, body: dict) -> Any:
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{SUPABASE_URL}/rest/v1/{path}",
                         headers=_sb_headers(), json=body)
    r.raise_for_status()
    return r.json()


async def _sb_rpc(fn: str, body: dict) -> Any:
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{SUPABASE_URL}/rest/v1/rpc/{fn}",
                         headers=_sb_headers(), json=body)
    r.raise_for_status()
    return r.json()


# ── Razorpay helpers ───────────────────────────────────────────────────────────

def _rzp_auth() -> tuple[str, str]:
    return (RZP_KEY_ID, RZP_KEY_SECRET)


async def _rzp_create_order(amount: int, receipt: str, notes: dict) -> dict:
    async with httpx.AsyncClient() as c:
        r = await c.post(
            f"{RZP_BASE}/orders",
            auth=_rzp_auth(),
            json={"amount": amount, "currency": "INR",
                  "receipt": receipt, "notes": notes},
        )
    if r.status_code != 200:
        raise HTTPException(502, f"Razorpay error: {r.text}")
    return r.json()


async def _rzp_fetch_payments(rp_order_id: str) -> list[dict]:
    """Return list of payment entities for a Razorpay order."""
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{RZP_BASE}/orders/{rp_order_id}/payments",
                        auth=_rzp_auth())
    if r.status_code != 200:
        return []
    return r.json().get("items", [])


# ── Endpoint 1: Create order ───────────────────────────────────────────────────

class OrderIn(BaseModel):
    plan_id:  str = "rookie_pass_yearly"
    user_id:  str          # UUID — sent by the Next.js server route after auth
    email:    str | None = None


@router.post("/orders")
async def create_order(body: OrderIn) -> dict:
    # 1. Fetch plan from Supabase
    plans = await _sb_get("plans", {"id": f"eq.{body.plan_id}", "select": "*"})
    if not plans:
        raise HTTPException(404, "Plan not found")
    plan = plans[0]

    # 2. Create Razorpay order
    order_id = str(uuid.uuid4())
    rp = await _rzp_create_order(
        amount=plan["price_paise"],
        receipt=order_id,
        notes={"user_id": body.user_id, "plan_id": body.plan_id},
    )

    # 3. Persist to Supabase orders table
    await _sb_post("orders", {
        "id":                 order_id,
        "user_id":            body.user_id,
        "plan_id":            body.plan_id,
        "razorpay_order_id":  rp["id"],
        "amount":             plan["price_paise"],
        "currency":           "INR",
        "status":             "created",
    })

    return {
        "order_id":  rp["id"],   # Razorpay order id (rzp_...)
        "key_id":    RZP_KEY_ID,
        "amount":    plan["price_paise"],
        "currency":  "INR",
    }


# ── Endpoint 2: Webhook — the ONLY thing that grants access ───────────────────

@router.post("/webhooks/razorpay")
async def razorpay_webhook(request: Request, bg: BackgroundTasks) -> dict:
    raw = await request.body()
    sig = request.headers.get("X-Razorpay-Signature", "")

    expected = hmac.new(
        WEBHOOK_SECRET.encode(), raw, hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected, sig):
        raise HTTPException(400, "Invalid signature")

    event = json.loads(raw)
    ev_type = event.get("event", "")

    if ev_type in ("order.paid", "payment.captured"):
        payment = event["payload"]["payment"]["entity"]
        bg.add_task(_settle, payment)

    return {"ok": True}


async def _settle(payment: dict) -> None:
    """Background: verify captured, then call grant_access()."""
    if payment.get("status") != "captured":
        return
    rp_order_id   = payment.get("order_id", "")
    rp_payment_id = payment.get("id", "")
    if not rp_order_id:
        return
    try:
        await _sb_rpc("grant_access", {
            "p_rp_order":   rp_order_id,
            "p_rp_payment": rp_payment_id,
        })
    except Exception as e:
        print(f"[settle] grant_access failed: {e}")


# ── Endpoint 3: Status (poll + self-heal) ─────────────────────────────────────

@router.get("/orders/{rp_order_id}/status")
async def order_status(rp_order_id: str, bg: BackgroundTasks) -> dict:
    """
    Returns our DB status for a Razorpay order.
    If still 'created' and Razorpay shows a captured payment,
    triggers settle() in the background so a delayed webhook is harmless.
    """
    rows = await _sb_get("orders", {
        "razorpay_order_id": f"eq.{rp_order_id}",
        "select": "status,paid_at",
    })
    if not rows:
        raise HTTPException(404, "Order not found")

    row = rows[0]
    if row["status"] == "created":
        # Self-heal: check Razorpay directly
        payments = await _rzp_fetch_payments(rp_order_id)
        for p in payments:
            if p.get("status") == "captured":
                bg.add_task(_settle, p)
                break

    return {"status": row["status"], "paid_at": row.get("paid_at")}
