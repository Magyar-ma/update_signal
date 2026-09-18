"""Incremental (stateful) indicator computations.

Each indicator exposes:
    - init(df): compute full history, return state dict
    - update(state, new_row): advance one candle, return updated series slice

State dict contains whatever is needed to resume from the last candle.
"""

import numpy as np
import pandas as pd


def _ema_state(series, period):
    ema = series.ewm(span=period, adjust=False, min_periods=period).mean()
    return ema.iloc[-1]


def _rma_state(series, period):
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().iloc[-1]


class EMAEngine:
    @staticmethod
    def init(df, periods=(20, 50, 100, 200)):
        state = {}
        for p in periods:
            state[p] = _ema_state(df["close"], p)
        return state

    @staticmethod
    def update(state, prev_close, new_close):
        out = {}
        for p, ema in state.items():
            alpha = 2.0 / (p + 1)
            new_ema = ema + alpha * (new_close - ema)
            out[p] = new_ema
            state[p] = new_ema
        return out


class ATRRMAEngine:
    @staticmethod
    def init(df, period=14):
        high, low, close = df["high"], df["low"], df["close"]
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = _rma_state(tr, period)
        return {"atr": atr, "prev_close": float(close.iloc[-1])}

    @staticmethod
    def update(state, prev_close, new_high, new_low, new_close, period=14):
        tr1 = new_high - new_low
        tr2 = abs(new_high - prev_close)
        tr3 = abs(new_low - prev_close)
        tr = max(tr1, tr2, tr3)
        atr = state["atr"] + (1.0 / period) * (tr - state["atr"])
        state["atr"] = atr
        state["prev_close"] = new_close
        return atr


class RSIEngine:
    @staticmethod
    def init(df, period=14):
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = _rma_state(gain, period)
        avg_loss = _rma_state(loss, period)
        return {"avg_gain": avg_gain, "avg_loss": avg_loss}

    @staticmethod
    def update(state, prev_close, new_close, period=14):
        change = new_close - prev_close
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = state["avg_gain"] + (1.0 / period) * (gain - state["avg_gain"])
        avg_loss = state["avg_loss"] + (1.0 / period) * (loss - state["avg_loss"])
        state["avg_gain"] = avg_gain
        state["avg_loss"] = avg_loss
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)


class OBVEngine:
    @staticmethod
    def init(df):
        close = df["close"]
        vol = df["volume"]
        direction = np.sign(close.diff())
        obv = (direction * vol).cumsum()
        return {"obv": float(obv.iloc[-1]), "prev_close": float(close.iloc[-1])}

    @staticmethod
    def update(state, prev_close, new_close, new_volume):
        if new_close > prev_close:
            obv = state["obv"] + new_volume
        elif new_close < prev_close:
            obv = state["obv"] - new_volume
        else:
            obv = state["obv"]
        state["obv"] = obv
        state["prev_close"] = new_close
        return obv


class CVDEngine:
    @staticmethod
    def init(df):
        high, low, close = df["high"], df["low"], df["close"]
        vol = df["volume"]
        candle_range = (high - low).replace(0, 1e-8)
        buy_ratio = (close - low) / candle_range
        sell_ratio = (high - close) / candle_range
        delta = vol * (buy_ratio - sell_ratio)
        cvd = delta.cumsum()
        return {"cvd": float(cvd.iloc[-1])}

    @staticmethod
    def update(state, new_high, new_low, new_close, new_volume):
        candle_range = max(new_high - new_low, 1e-8)
        buy_ratio = (new_close - new_low) / candle_range
        sell_ratio = (new_high - new_close) / candle_range
        delta = new_volume * (buy_ratio - sell_ratio)
        cvd = state["cvd"] + delta
        state["cvd"] = cvd
        return cvd


class VWAPEngine:
    @staticmethod
    def init(df):
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        tp_vol = typical * df["volume"]
        cum_tp = tp_vol.cumsum()
        cum_vol = df["volume"].cumsum()
        vwap = cum_tp.iloc[-1] / cum_vol.iloc[-1]
        return {"cum_tp": float(cum_tp.iloc[-1]), "cum_vol": float(cum_vol.iloc[-1])}

    @staticmethod
    def update(state, new_high, new_low, new_close, new_volume):
        typical = (new_high + new_low + new_close) / 3.0
        state["cum_tp"] += typical * new_volume
        state["cum_vol"] += new_volume
        return state["cum_tp"] / state["cum_vol"] if state["cum_vol"] > 0 else np.nan
