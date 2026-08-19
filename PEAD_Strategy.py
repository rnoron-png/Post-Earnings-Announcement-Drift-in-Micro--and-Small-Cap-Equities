"""
Post-Earnings-Announcement Drift Strategy for U.S. Micro- and Small-Cap Equities.

A long-only implementation of the post-earnings-announcement-drift (PEAD)
anomaly on the QuantConnect LEAN engine. Each trading day the strategy admits
recently-reporting micro- and small-cap names, ranks them by a benchmark-
adjusted price-reaction surprise proxy, and holds the top decile for a fixed
horizon. Position sizing is liquidity-aware, risk is managed with hard and
trailing stops, and idle capital may optionally be swept into a Treasury-bill
ETF. All returns are net of the platform's default commission model.

Two reported configurations differ only in the treatment of idle capital:
    Uninvested      surplus cash is held as cash.
    Treasury-swept  surplus cash is swept into BIL.

Development window: 2015-2024. The period from 2025 onward is reserved as an
untouched out-of-sample holdout.
"""

from AlgorithmImports import *
import numpy as np


class MicroCapEarningsUniverseSelectionModel(FundamentalUniverseSelectionModel):
    """Selects micro- and small-cap securities that have reported earnings
    within a short trailing window, subject to price and liquidity floors,
    capped at the most liquid names by dollar volume.
    """

    def __init__(self, min_price, min_dollar_volume, min_market_cap, max_market_cap,
                 lookback_days_for_earnings=3, max_universe_size=250):
        super().__init__(True)
        self._min_price = min_price
        self._min_dollar_volume = min_dollar_volume
        self._min_market_cap = min_market_cap
        self._max_market_cap = max_market_cap
        self._lookback_days_for_earnings = lookback_days_for_earnings
        self._max_universe_size = max_universe_size

    def select(self, algorithm, fundamental):
        filtered = [
            f for f in fundamental
            if f.has_fundamental_data
            and f.price > self._min_price
            and f.dollar_volume > self._min_dollar_volume
            and self._min_market_cap < f.market_cap < self._max_market_cap
        ]

        recent_reporters = []
        for f in filtered:
            report_date = self._get_earnings_report_date(f)
            if report_date is None:
                continue
            days_since = (algorithm.time.date() - report_date).days
            if 0 <= days_since <= self._lookback_days_for_earnings:
                recent_reporters.append(f)

        recent_reporters.sort(key=lambda f: f.dollar_volume, reverse=True)
        return [f.symbol for f in recent_reporters[:self._max_universe_size]]

    def _get_earnings_report_date(self, fundamental):
        # Filing date; for small issuers this may lag the earnings press release.
        earning_reports = fundamental.earning_reports
        if earning_reports is None:
            return None
        file_date = earning_reports.file_date
        if file_date is None or file_date.value is None:
            return None
        return file_date.value.date()


class PEADAlphaModel(AlphaModel):
    """Generates long insights from a benchmark-adjusted price-reaction proxy.

    For each candidate the proxy is the security's cumulative return over the
    event window minus the contemporaneous return of the micro-cap benchmark
    (IWC). Candidates are ranked against a rolling population of past proxy
    values; those in the top decile receive an upward insight held for a fixed
    horizon.
    """

    def __init__(self, algorithm, holding_period_days, decile_count, parking_symbol,
                 min_history_for_ranking=20, max_history_size=1000):
        self._holding_period_days = holding_period_days
        self._decile_count = decile_count
        self._parking_symbol = parking_symbol
        self._min_history_for_ranking = min_history_for_ranking
        self._max_history_size = max_history_size
        self.benchmark_symbol = algorithm.add_equity("IWC", Resolution.DAILY).symbol
        self.event_windows = {}
        self.proxy_history = []

    def on_securities_changed(self, algorithm, changes):
        for security in changes.added_securities:
            if security.symbol == self.benchmark_symbol:
                continue
            if security.symbol == self._parking_symbol:
                continue
            self.event_windows[security.symbol] = RollingWindow[float](5)
        for security in changes.removed_securities:
            if security.symbol in self.event_windows:
                del self.event_windows[security.symbol]

    def update(self, algorithm, data):
        insights = []
        candidates = []

        for symbol, window in self.event_windows.items():
            if not data.bars.contains_key(symbol):
                continue
            window.add(data.bars[symbol].close)
            if window.count < 3:
                continue
            proxy = self._compute_surprise_proxy(algorithm, window)
            if proxy is None:
                continue
            candidates.append((symbol, proxy))

        if not candidates:
            return insights

        long_cutoff = None
        if len(self.proxy_history) >= self._min_history_for_ranking:
            top_pct = 100.0 * (1.0 - 1.0 / self._decile_count)
            long_cutoff = float(np.percentile(self.proxy_history, top_pct))

        duration = timedelta(days=self._holding_period_days)

        if long_cutoff is not None:
            passers = [(s, p) for s, p in candidates if p >= long_cutoff]
            for symbol, proxy in passers:
                insights.append(Insight.price(
                    symbol, duration, InsightDirection.UP,
                    magnitude=abs(proxy), confidence=self._confidence(proxy)
                ))

        self.proxy_history.extend(proxy for _, proxy in candidates)
        if len(self.proxy_history) > self._max_history_size:
            self.proxy_history = self.proxy_history[-self._max_history_size:]

        return insights

    def _compute_surprise_proxy(self, algorithm, window):
        # Cumulative return of the name over the event window, minus the
        # benchmark's return over the same window (a benchmark-adjusted CAR).
        if not algorithm.securities.contains_key(self.benchmark_symbol):
            return None

        prices = [window[i] for i in range(window.count)][::-1]
        if len(prices) < 2 or prices[0] == 0:
            return None
        stock_return = (prices[-1] / prices[0]) - 1.0

        benchmark_history = algorithm.history(self.benchmark_symbol, len(prices), Resolution.DAILY)
        if benchmark_history.empty:
            return None
        if "close" not in benchmark_history.columns:
            return None
        bench_closes = benchmark_history["close"].values
        if len(bench_closes) < 2 or bench_closes[0] == 0:
            return None
        benchmark_return = (bench_closes[-1] / bench_closes[0]) - 1.0

        return stock_return - benchmark_return

    def _confidence(self, proxy):
        # Recorded on the insight but not used in position sizing.
        return float(1.0 / (1.0 + np.exp(-8 * abs(proxy))))


class LiquidityAwarePortfolioConstructionModel(PortfolioConstructionModel):
    """Equal-weights active insights subject to a per-name cap and a
    participation cap tied to each name's average dollar volume. Expired-
    insight positions are liquidated each cycle, and surplus capital may be
    swept into a Treasury-bill ETF.
    """

    def __init__(self, max_single_name_weight, max_gross_exposure,
                 max_position_pct_of_adv, stop_registry, stop_cooldown_days,
                 parking_symbol, use_tbill_sweep, cash_target_cap=0.60):
        self._max_single_name_weight = max_single_name_weight
        self._max_gross_exposure = max_gross_exposure
        self._max_position_pct_of_adv = max_position_pct_of_adv
        self._stop_registry = stop_registry
        self._stop_cooldown_days = stop_cooldown_days
        self._parking_symbol = parking_symbol
        self._cash_target_cap = cash_target_cap
        self._use_tbill_sweep = use_tbill_sweep
        self._previously_targeted = set()
        self.insight_collection = InsightCollection()

    def create_targets(self, algorithm, insights):
        for insight in insights:
            self.insight_collection.add(insight)

        active_insights = self.insight_collection.get_active_insights(algorithm.utc_time)
        if not active_insights:
            return self._with_parking(algorithm, [], 0.0)

        # A stopped symbol stays out for the remainder of its holding window.
        today = algorithm.time.date()
        eligible = []
        for insight in active_insights:
            stop_date = self._stop_registry.get(insight.symbol)
            if stop_date is not None and (today - stop_date).days < self._stop_cooldown_days:
                continue
            eligible.append(insight)
        if not eligible:
            return self._with_parking(algorithm, [], 0.0)

        n = len(eligible)
        equal_weight = min(1.0 / n, self._max_single_name_weight)

        weights = {}
        for insight in eligible:
            direction = 1 if insight.direction == InsightDirection.UP else -1
            weights[insight.symbol] = direction * equal_weight

        gross = sum(abs(w) for w in weights.values())
        if gross > self._max_gross_exposure:
            scale = self._max_gross_exposure / gross
            weights = {s: w * scale for s, w in weights.items()}

        targets = []
        signal_gross = 0.0
        total_value = algorithm.portfolio.total_portfolio_value

        for symbol, weight in weights.items():
            adv_dollars = self._get_20d_adv_dollars(algorithm, symbol)
            if not adv_dollars or adv_dollars <= 0:
                adv_dollars = self._get_same_day_dollar_volume_fallback(algorithm, symbol)
            if not adv_dollars or adv_dollars <= 0:
                continue

            max_dollars_by_liquidity = adv_dollars * self._max_position_pct_of_adv
            target_dollars = total_value * weight
            capped_dollars = min(abs(target_dollars), max_dollars_by_liquidity)
            capped_dollars *= 1 if target_dollars >= 0 else -1
            capped_weight = capped_dollars / total_value if total_value > 0 else 0.0

            target = PortfolioTarget.percent(algorithm, symbol, capped_weight)
            if target is not None:
                targets.append(target)
                signal_gross += abs(capped_weight)

        return self._with_parking(algorithm, targets, signal_gross)

    def _with_parking(self, algorithm, targets, signal_gross):
        # Liquidate any previously-held name no longer being targeted, so
        # expired-insight positions do not persist.
        current_symbols = {t.symbol for t in targets}
        for stale_symbol in list(self._previously_targeted):
            if stale_symbol in current_symbols:
                continue
            if algorithm.portfolio[stale_symbol].invested:
                flat = PortfolioTarget.percent(algorithm, stale_symbol, 0.0)
                if flat is not None:
                    targets.append(flat)
        self._previously_targeted = current_symbols

        # The Treasury-bill sweep is the only difference between the two
        # reported configurations.
        if not self._use_tbill_sweep:
            return targets

        parking_weight = max(0.0, self._cash_target_cap - signal_gross)
        equity = float(algorithm.portfolio.total_portfolio_value)
        current_parking = (float(algorithm.portfolio[self._parking_symbol].holdings_value) / equity
                           if equity > 0 else 0.0)
        # Rebalance the sweep only on material drift, to avoid daily churn.
        if abs(parking_weight - current_parking) >= 0.05:
            parking_target = PortfolioTarget.percent(algorithm, self._parking_symbol, parking_weight)
            if parking_target is not None:
                targets.append(parking_target)
        return targets

    def _get_20d_adv_dollars(self, algorithm, symbol):
        history = algorithm.history(symbol, 20, Resolution.DAILY)
        if history.empty:
            return None
        if "close" not in history.columns or "volume" not in history.columns:
            return None
        return float((history["close"] * history["volume"]).mean())

    def _get_same_day_dollar_volume_fallback(self, algorithm, symbol):
        if not algorithm.securities.contains_key(symbol):
            return None
        security = algorithm.securities[symbol]
        if security.volume <= 0 or security.price <= 0:
            return None
        return float(security.volume * security.price)


class PEADRiskManagementModel(RiskManagementModel):
    """Applies a hard stop loss and a trailing-giveback stop to each position.
    On a stop, the position is flattened, its symbol is registered for a
    cooldown, and its insight is cancelled so it is not re-entered.
    """

    def __init__(self, stop_registry, parking_symbol,
                 stop_loss_pct=0.15, trailing_giveback_pct=0.10):
        self._stop_registry = stop_registry
        self._parking_symbol = parking_symbol
        self._stop_loss_pct = stop_loss_pct
        self._trailing_giveback_pct = trailing_giveback_pct
        self.entry_prices = {}
        self.extreme_prices = {}

    def manage_risk(self, algorithm, targets):
        risk_targets = []

        for kvp in algorithm.portfolio:
            symbol = kvp.key
            holding = kvp.value

            if symbol == self._parking_symbol:
                continue

            if not holding.invested:
                if symbol in self.entry_prices:
                    del self.entry_prices[symbol]
                if symbol in self.extreme_prices:
                    del self.extreme_prices[symbol]
                continue

            price = holding.price
            is_long = holding.is_long

            if symbol not in self.entry_prices:
                self.entry_prices[symbol] = holding.average_price
                self.extreme_prices[symbol] = price

            if is_long:
                self.extreme_prices[symbol] = max(self.extreme_prices[symbol], price)
            else:
                self.extreme_prices[symbol] = min(self.extreme_prices[symbol], price)

            entry = self.entry_prices[symbol]
            extreme = self.extreme_prices[symbol]

            hard_stop_hit = (
                (is_long and price <= entry * (1 - self._stop_loss_pct)) or
                (not is_long and price >= entry * (1 + self._stop_loss_pct))
            )

            giveback = abs(extreme - price) / extreme if extreme != 0 else 0
            trailing_stop_hit = giveback >= self._trailing_giveback_pct and (
                (is_long and price < extreme) or (not is_long and price > extreme)
            )

            if hard_stop_hit or trailing_stop_hit:
                risk_targets.append(PortfolioTarget(symbol, 0))
                self._stop_registry[symbol] = algorithm.time.date()
                algorithm.insights.cancel([symbol])
                del self.entry_prices[symbol]
                del self.extreme_prices[symbol]

        return risk_targets


class SpreadAwareExecutionModel(ExecutionModel):
    """Executes targets with a spread gate and sells-before-buys ordering.

    Entries are placed as limit orders priced a quarter of the way into the
    spread, skipped when the spread is too wide or margin is insufficient.
    Sells are sequenced ahead of buys so freed capital funds new entries in
    the same cycle. The Treasury-bill sweep trades at market.
    """

    def __init__(self, max_spread_pct, parking_symbol):
        self._max_spread_pct = max_spread_pct
        self._parking_symbol = parking_symbol
        self.targets_collection = PortfolioTargetCollection()

    def execute(self, algorithm, targets):
        self.targets_collection.add_range(targets)
        if not self.targets_collection.count:
            return

        pending = []
        for target in self.targets_collection.order_by_margin_impact(algorithm):
            symbol = target.symbol
            # Never stack on top of an open order.
            if len(algorithm.transactions.get_open_orders(symbol)) > 0:
                continue
            existing_quantity = algorithm.portfolio[symbol].quantity
            target_quantity = target.quantity - existing_quantity
            if target_quantity == 0:
                continue
            pending.append((float(target_quantity), symbol))

        # Sells (negative quantity) first, so their proceeds fund buys.
        pending.sort(key=lambda item: item[0])

        for target_quantity, symbol in pending:
            security = algorithm.securities[symbol]

            if symbol == self._parking_symbol:
                algorithm.market_order(symbol, target_quantity)
                continue

            # Defer a buy that would exceed available margin rather than
            # submitting an order that would be rejected and resubmitted.
            if target_quantity > 0:
                estimated_cost = abs(target_quantity) * float(security.price)
                if estimated_cost > float(algorithm.portfolio.margin_remaining) * 0.95:
                    continue

            bid = security.bid_price
            ask = security.ask_price
            if bid <= 0 or ask <= 0:
                continue

            spread_pct = (ask - bid) / ((ask + bid) / 2)
            if spread_pct > self._max_spread_pct:
                continue

            if target_quantity > 0:
                limit_price = round(bid + 0.25 * (ask - bid), 2)
            else:
                limit_price = round(ask - 0.25 * (ask - bid), 2)
            algorithm.limit_order(symbol, target_quantity, limit_price)

        self.targets_collection.clear_fulfilled(algorithm)


class MicroCapPEADAlgorithm(QCAlgorithm):
    """Long-only PEAD strategy for U.S. micro- and small-cap equities.

    Set ``use_tbill_sweep`` to select between the two reported configurations:
    True sweeps idle capital into BIL, False leaves it uninvested.
    """

    def initialize(self):
        self.set_start_date(2015, 1, 1)
        self.set_end_date(2024, 12, 31)      # 2025+ reserved as out-of-sample holdout
        self.set_cash(50_000)

        self.settings.free_portfolio_value_percentage = 0.05
        self.universe_settings.resolution = Resolution.DAILY
        self.universe_settings.leverage = 1.0

        # Universe filters.
        min_market_cap = 50_000_000
        max_market_cap = 2_000_000_000
        min_price = 2.0
        min_dollar_volume = 300_000

        # Signal and portfolio parameters.
        holding_period_days = 40
        decile_count = 10
        max_single_name_weight = 0.06
        max_position_pct_of_adv = 0.01
        max_gross_exposure = 0.85
        max_spread_pct = 0.03

        # Configuration switch: idle cash to T-bills (True) or held as cash (False).
        use_tbill_sweep = True

        stop_registry = {}

        parking_equity = self.add_equity("BIL", Resolution.DAILY)
        parking_equity.set_leverage(1.0)
        parking_symbol = parking_equity.symbol
        self._parking_symbol = parking_symbol

        self.add_universe_selection(MicroCapEarningsUniverseSelectionModel(
            min_price=min_price,
            min_dollar_volume=min_dollar_volume,
            min_market_cap=min_market_cap,
            max_market_cap=max_market_cap,
        ))
        self.add_alpha(PEADAlphaModel(
            self,
            holding_period_days=holding_period_days,
            decile_count=decile_count,
            parking_symbol=parking_symbol,
        ))
        self.set_portfolio_construction(LiquidityAwarePortfolioConstructionModel(
            max_single_name_weight=max_single_name_weight,
            max_gross_exposure=max_gross_exposure,
            max_position_pct_of_adv=max_position_pct_of_adv,
            stop_registry=stop_registry,
            stop_cooldown_days=holding_period_days,
            parking_symbol=parking_symbol,
            use_tbill_sweep=use_tbill_sweep,
        ))
        self.add_risk_management(PEADRiskManagementModel(
            stop_registry=stop_registry,
            parking_symbol=parking_symbol,
        ))
        self.set_execution(SpreadAwareExecutionModel(
            max_spread_pct=max_spread_pct,
            parking_symbol=parking_symbol,
        ))

        self.set_warm_up(timedelta(days=90))

        # Deployment accounting: daily gross exposure of signal positions,
        # excluding the Treasury-bill sweep.
        self._gross_samples = []
        self.schedule.on(self.date_rules.every_day(),
                         self.time_rules.at(15, 45),
                         self._record_gross)

    def _record_gross(self):
        if self.is_warming_up:
            return
        total_value = float(self.portfolio.total_portfolio_value)
        if total_value <= 0:
            return
        gross = sum(abs(float(kvp.value.holdings_value)) for kvp in self.portfolio
                    if kvp.key != self._parking_symbol) / total_value
        self._gross_samples.append(gross)

    def on_end_of_algorithm(self):
        final_equity = float(self.portfolio.total_portfolio_value)
        if len(self._gross_samples) > 0:
            avg_gross = sum(self._gross_samples) / len(self._gross_samples)
            max_gross = max(self._gross_samples)
        else:
            avg_gross = 0.0
            max_gross = 0.0
        years = 10.0
        headline_cagr = (final_equity / 50_000) ** (1 / years) - 1 if final_equity > 0 else -1
        return_on_deployed = headline_cagr / avg_gross if avg_gross > 0 else 0.0

        self.set_runtime_statistic("CAGR", f"{headline_cagr:.2%}")
        self.set_runtime_statistic("Avg Deployment", f"{avg_gross:.1%}")
        self.set_runtime_statistic("Max Deployment", f"{max_gross:.1%}")
        self.set_runtime_statistic("Return on Deployed", f"{return_on_deployed:.2%}")
        self.set_runtime_statistic("Fees", f"{float(self.portfolio.total_fees):,.0f}")
