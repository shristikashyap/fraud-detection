import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "./api.js";

const NAV_ITEMS = [
  { id: "overview", label: "Overview", glyph: "OV" },
  { id: "transactions", label: "Transaction monitoring", glyph: "TX" },
  { id: "behavior", label: "Behaviour analysis", glyph: "BH" },
  { id: "detection", label: "Fraud detection", glyph: "ML" },
  { id: "risk", label: "Risk analysis", glyph: "RS" },
  { id: "alerts", label: "Alerts & notifications", glyph: "AL" },
  { id: "reports", label: "Reports", glyph: "RP" },
];

const RISK_COLORS = {
  "LOW RISK": "#31c48d",
  "MEDIUM RISK": "#f4a340",
  "HIGH RISK": "#f06a73",
};

const PAGE_TITLES = {
  overview: ["Overview", "Live view of the risk monitoring service."],
  transactions: ["Transaction monitoring", "Analyze activity and review persistent transaction history."],
  behavior: ["Behaviour analysis", "Trusted user baselines and contextual deviations."],
  detection: ["Fraud detection", "Separate outputs from the transaction classifier and anomaly models."],
  risk: ["Risk analysis", "Review the combined triage score and its component contributions."],
  alerts: ["Alerts & notifications", "Activities recommended for additional verification."],
  reports: ["Reports", "Actual backend summaries and current-session transaction totals."],
};

const EMPTY_REPORT = null;
const TRANSACTION_PAGE_SIZE = 20;
const ALERT_PAGE_SIZE = 50;

function formatNumber(value, digits = 0) {
  return Number.isFinite(value)
    ? new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value)
    : "—";
}

function formatMoney(value) {
  return Number.isFinite(value)
    ? new Intl.NumberFormat(undefined, {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 2,
      }).format(value)
    : "—";
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? String(value)
    : date.toLocaleString(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      });
}

function riskClass(category) {
  if (category === "LOW RISK") return "risk-low";
  if (category === "MEDIUM RISK") return "risk-medium";
  if (category === "HIGH RISK") return "risk-high";
  return "risk-neutral";
}

function modelStatusLabel(model) {
  if (!model) return "Status unavailable";
  if (model.available) return "Available";
  if (model.status === "missing") return "Artifact missing";
  if (model.status === "load_error") return "Load failed";
  return "Unavailable";
}

function App() {
  const [activePage, setActivePage] = useState("overview");
  const [health, setHealth] = useState(null);
  const [report, setReport] = useState(EMPTY_REPORT);
  const [alerts, setAlerts] = useState([]);
  const [alertCount, setAlertCount] = useState(null);
  const [alertPage, setAlertPage] = useState(1);
  const [alertTotalPages, setAlertTotalPages] = useState(0);
  const [alertsAvailable, setAlertsAvailable] = useState(false);
  const [users, setUsers] = useState([]);
  const [transactions, setTransactions] = useState([]);
  const [transactionCount, setTransactionCount] = useState(null);
  const [transactionPage, setTransactionPage] = useState(1);
  const [transactionTotalPages, setTransactionTotalPages] = useState(0);
  const [transactionsAvailable, setTransactionsAvailable] = useState(false);
  const [sessionAnalyses, setSessionAnalyses] = useState([]);
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [apiError, setApiError] = useState("");
  const [submitError, setSubmitError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);

  const refreshDashboard = useCallback(async () => {
    setRefreshing(true);
    const results = await Promise.allSettled([
      api.health(),
      api.reports(),
      api.alerts(alertPage, ALERT_PAGE_SIZE),
      api.users(),
      api.transactions(transactionPage, TRANSACTION_PAGE_SIZE),
    ]);
    const labels = ["health", "reports", "alerts", "user profiles", "transactions"];
    const failures = [];

    if (results[0].status === "fulfilled") {
      setHealth(results[0].value);
    } else {
      failures.push(`${labels[0]}: ${results[0].reason.message}`);
    }
    if (results[1].status === "fulfilled") {
      setReport(results[1].value);
    } else {
      failures.push(`${labels[1]}: ${results[1].reason.message}`);
    }
    if (results[2].status === "fulfilled") {
      setAlerts(results[2].value.alerts ?? []);
      setAlertCount(results[2].value.count ?? 0);
      setAlertTotalPages(results[2].value.total_pages ?? 0);
      setAlertsAvailable(true);
    } else {
      failures.push(`${labels[2]}: ${results[2].reason.message}`);
      setAlertsAvailable(false);
    }
    if (results[3].status === "fulfilled") {
      setUsers(results[3].value.users ?? []);
    } else {
      failures.push(`${labels[3]}: ${results[3].reason.message}`);
    }
    if (results[4].status === "fulfilled") {
      setTransactions(results[4].value.transactions ?? []);
      setTransactionCount(results[4].value.count ?? 0);
      setTransactionTotalPages(results[4].value.total_pages ?? 0);
      setTransactionsAvailable(true);
    } else {
      failures.push(`${labels[4]}: ${results[4].reason.message}`);
      setTransactionsAvailable(false);
    }
    setApiError(
      failures.length
        ? `Some live data could not be loaded. ${failures.join(" · ")}`
        : "",
    );
    setLastUpdated(new Date().toISOString());
    setRefreshing(false);
  }, [alertPage, transactionPage]);

  useEffect(() => {
    refreshDashboard();
  }, [refreshDashboard]);

  const handleAnalyze = async (payload, idempotencyKey) => {
    setSubmitting(true);
    setSubmitError("");
    try {
      const result = await api.analyze(payload, idempotencyKey);
      setSelectedAnalysis(result);
      setSessionAnalyses((current) => [result, ...current].slice(0, 100));
      await refreshDashboard();
      return result;
    } catch (error) {
      setSubmitError(error.message);
      return null;
    } finally {
      setSubmitting(false);
    }
  };

  const pageInfo = PAGE_TITLES[activePage];
  const activeAlertCount = alertCount ?? alerts.length;
  const backendAvailable = health?.backend_available === true;
  const databaseAvailable = health?.database?.available === true;

  return (
    <div className="dashboard-shell">
      <aside className="sidebar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 32 32" fill="none">
              <path
                d="M16 3.5 26 7v7.6c0 6.3-4.1 11.6-10 14-5.9-2.4-10-7.7-10-14V7l10-3.5Z"
                stroke="currentColor"
                strokeWidth="1.8"
              />
              <path d="m11 16 3.1 3.1L21 12" stroke="currentColor" strokeWidth="2" />
            </svg>
          </div>
          <div>
            <div className="brand-name">SENTINEL</div>
            <div className="brand-caption">RISK INTELLIGENCE</div>
          </div>
        </div>

        <div className="nav-section-label">WORKSPACE</div>
        <nav className="side-nav" aria-label="Dashboard sections">
          {NAV_ITEMS.map((item) => (
            <button
              className={`nav-item ${activePage === item.id ? "active" : ""}`}
              key={item.id}
              onClick={() => setActivePage(item.id)}
              type="button"
              aria-current={activePage === item.id ? "page" : undefined}
            >
              <span className="nav-glyph" aria-hidden="true">
                {item.glyph}
              </span>
              <span>{item.label}</span>
              {item.id === "alerts" && activeAlertCount > 0 && (
                <span className="nav-count">{activeAlertCount}</span>
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-spacer" />
        <div className="sidebar-status">
          <span className={`status-led ${backendAvailable ? "online" : ""}`} />
          <div>
            <div className="sidebar-status-title">
              {backendAvailable ? "API available" : "API status"}
            </div>
            <div className="sidebar-status-sub">
              {databaseAvailable ? "MongoDB connected" : health?.database?.status ?? "Checking service"}
            </div>
          </div>
        </div>
        <div className="sidebar-footer">CAPSTONE RESEARCH · PERSISTENT STORAGE</div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">
            <span>Workspace</span>
            <span className="breadcrumb-separator">/</span>
            <strong>{pageInfo[0]}</strong>
          </div>
          <div className="topbar-actions">
            <span className={`service-pill ${databaseAvailable ? "connected" : "disconnected"}`}>
              <span className="status-led" />
              {!health ? "Checking service" : databaseAvailable ? "Systems operational" : backendAvailable ? "MongoDB unavailable" : "Systems unavailable"}
            </span>
            <button
              className="icon-button refresh-button"
              type="button"
              onClick={refreshDashboard}
              disabled={refreshing}
              aria-label="Refresh dashboard data"
              title="Refresh dashboard data"
            >
              <span className={refreshing ? "refresh-symbol spinning" : "refresh-symbol"}>↻</span>
            </button>
          </div>
        </header>

        <div className="page-content">
          <div className="page-heading">
            <div>
              <p className="eyebrow">CONTEXT-AWARE FRAUD MONITORING</p>
              <h1>{pageInfo[0]}</h1>
              <p className="page-subtitle">{pageInfo[1]}</p>
            </div>
            <div className="last-updated">
              <span className="last-updated-label">LAST SYNC</span>
              <span>{lastUpdated ? formatDate(lastUpdated) : "—"}</span>
            </div>
          </div>

          {apiError && (
            <div className="message-banner error-banner" role="alert">
              <span className="message-icon">!</span>
              <span>{apiError}</span>
              <button type="button" className="message-close" onClick={() => setApiError("")} aria-label="Dismiss error">
                ×
              </button>
            </div>
          )}

          {activePage === "overview" && (
            <OverviewPage
              report={report}
              alerts={alerts}
              alertsAvailable={alertsAvailable}
              health={health}
              refreshing={refreshing}
              onNavigate={setActivePage}
              onSelectAlert={() => setActivePage("alerts")}
            />
          )}
          {activePage === "transactions" && (
            <TransactionPage
              analyses={transactions}
              selectedAnalysis={selectedAnalysis}
              historyCount={transactionCount}
              historyPage={transactionPage}
              historyTotalPages={transactionTotalPages}
              historyAvailable={transactionsAvailable}
              historyLoading={refreshing}
              onHistoryPageChange={setTransactionPage}
              submitting={submitting}
              submitError={submitError}
              onAnalyze={handleAnalyze}
              onOpenResult={(analysis) => {
                setSelectedAnalysis(analysis);
                setActivePage("risk");
              }}
            />
          )}
          {activePage === "behavior" && (
            <BehaviorPage users={users} latestAnalysis={selectedAnalysis} />
          )}
          {activePage === "detection" && (
            <DetectionPage analysis={selectedAnalysis} />
          )}
          {activePage === "risk" && (
            <RiskPage analysis={selectedAnalysis} />
          )}
          {activePage === "alerts" && (
            <AlertsPage
              alerts={alerts}
              count={alertCount}
              page={alertPage}
              totalPages={alertTotalPages}
              available={alertsAvailable}
              loading={refreshing}
              onPageChange={setAlertPage}
            />
          )}
          {activePage === "reports" && (
            <ReportsPage
              report={report}
              analyses={sessionAnalyses}
              refreshing={refreshing}
            />
          )}

          <footer className="content-footer">
            <span>Research prototype · scores are triage guidance, not confirmed fraud findings.</span>
            <span>Behavioral models use synthetic research data.</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

function OverviewPage({
  report,
  alerts,
  alertsAvailable,
  health,
  refreshing,
  onNavigate,
  onSelectAlert,
}) {
  const counts = report?.risk_category_counts;
  const riskData = counts
    ? Object.entries(RISK_COLORS).map(([name, color]) => ({
        name,
        value: counts[name] ?? 0,
        color,
      }))
    : [];
  const hasRiskData = riskData.some((item) => item.value > 0);

  return (
    <div className="page-stack">
      <section className="metric-grid">
        <MetricCard
          label="Analyzed activities"
          value={report ? formatNumber(report.assessments_total) : "—"}
          detail="Persisted by the backend"
          glyph="Σ"
          tone="teal"
        />
        <MetricCard
          label="Low risk"
          value={counts ? formatNumber(counts["LOW RISK"]) : "—"}
          detail="Below verification threshold"
          glyph="L"
          tone="green"
        />
        <MetricCard
          label="Medium risk"
          value={counts ? formatNumber(counts["MEDIUM RISK"]) : "—"}
          detail="Additional verification suggested"
          glyph="M"
          tone="orange"
        />
        <MetricCard
          label="High risk"
          value={counts ? formatNumber(counts["HIGH RISK"]) : "—"}
          detail="Priority review suggested"
          glyph="H"
          tone="red"
        />
      </section>

      <section className="overview-grid">
        <Panel
          title="Risk distribution"
          subtitle="Persisted assessments returned by the backend"
          action={<span className="panel-period">ALL SAVED RECORDS</span>}
        >
          {hasRiskData ? (
            <div className="chart-layout">
              <div className="chart-wrap donut-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={riskData}
                      dataKey="value"
                      nameKey="name"
                      innerRadius={66}
                      outerRadius={94}
                      paddingAngle={4}
                      stroke="none"
                    >
                      {riskData.map((entry) => (
                        <Cell key={entry.name} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip content={<ChartTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
                <div className="donut-center">
                  <strong>{formatNumber(report.assessments_total)}</strong>
                  <span>ANALYSES</span>
                </div>
              </div>
              <div className="chart-legend">
                {riskData.map((item) => (
                  <div className="legend-row" key={item.name}>
                    <span className="legend-dot" style={{ background: item.color }} />
                    <span>{item.name}</span>
                    <strong>{formatNumber(item.value)}</strong>
                  </div>
                ))}
                {report.unavailable_assessments > 0 && (
                  <div className="legend-row muted">
                    <span className="legend-dot unavailable-dot" />
                    <span>Unavailable score</span>
                    <strong>{formatNumber(report.unavailable_assessments)}</strong>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <EmptyState
              title={refreshing ? "Loading risk data" : "No assessments recorded"}
              description="Risk distribution will appear after real analyses are submitted."
            />
          )}
        </Panel>

        <Panel
          title="Model availability"
          subtitle="Loaded model artifacts reported by Flask"
          action={<span className={`small-status ${health?.status === "ok" ? "text-green" : "text-orange"}`}>{health?.status ?? "UNKNOWN"}</span>}
        >
          <ModelAvailability models={health?.models} database={health?.database} />
        </Panel>
      </section>

      <Panel
        title="Recent review alerts"
        subtitle="Anomaly-based review recommendations; not confirmed fraud"
        action={
          <button className="text-button" type="button" onClick={onSelectAlert}>
            View all <span aria-hidden="true">→</span>
          </button>
        }
      >
        {alerts.length ? (
          <div className="alert-list compact-alerts">
            {alerts.slice(0, 4).map((alert) => (
              <AlertRow alert={alert} key={alert.alert_id} onClick={onSelectAlert} />
            ))}
          </div>
        ) : (
          <EmptyState
            title={
              refreshing
                ? "Loading alerts"
                : alertsAvailable
                  ? "No review alerts"
                  : "Persistent alerts unavailable"
            }
            description={
              alertsAvailable
                ? "No saved assessment currently requires additional verification."
                : "Check MongoDB connectivity and refresh the dashboard."
            }
            action={
              <button className="button button-secondary" type="button" onClick={() => onNavigate("transactions")}>
                Analyze an activity
              </button>
            }
          />
        )}
      </Panel>
    </div>
  );
}

function TransactionPage({
  analyses,
  selectedAnalysis,
  historyCount,
  historyPage,
  historyTotalPages,
  historyAvailable,
  historyLoading,
  onHistoryPageChange,
  submitting,
  submitError,
  onAnalyze,
  onOpenResult,
}) {
  const [form, setForm] = useState(createInitialForm);
  const idempotencyKey = useRef(null);
  const [historyError, setHistoryError] = useState("");
  const [search, setSearch] = useState("");
  const [riskFilter, setRiskFilter] = useState("ALL");

  const filteredAnalyses = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return analyses.filter((item) => {
      const category = item.risk_assessment?.risk_category ?? "UNAVAILABLE";
      const matchesRisk = riskFilter === "ALL" || category === riskFilter;
      const matchesSearch =
        !needle ||
        item.user_id?.toLowerCase().includes(needle) ||
        item.transaction?.type?.toLowerCase().includes(needle);
      return matchesRisk && matchesSearch;
    });
  }, [analyses, riskFilter, search]);

  const update = (event) => {
    const { name, value } = event.target;
    setForm((current) => ({ ...current, [name]: value }));
    idempotencyKey.current = globalThis.crypto?.randomUUID?.() ?? null;
  };

  const submit = async (event) => {
    event.preventDefault();
    setHistoryError("");
    let history;
    try {
      history = form.recentActivities.trim()
        ? JSON.parse(form.recentActivities)
        : [];
      if (!Array.isArray(history)) {
        throw new Error("Prior activity must be a JSON array.");
      }
    } catch (error) {
      setHistoryError(error.message || "Prior activity must be valid JSON.");
      return;
    }

    const amount =
      form.transactionAmount.trim() === "" ? null : Number(form.transactionAmount);
    const frequency =
      form.transactionFrequency.trim() === "" ? null : Number(form.transactionFrequency);
    if (amount !== null && (!Number.isFinite(amount) || amount < 0)) {
      setHistoryError("Transaction amount must be a non-negative number.");
      return;
    }
    if (
      frequency !== null &&
      (!Number.isInteger(frequency) || frequency < 0)
    ) {
      setHistoryError("Transaction frequency must be a non-negative whole number.");
      return;
    }
    const localDate = new Date(form.timestamp);
    if (Number.isNaN(localDate.getTime())) {
      setHistoryError("Enter a valid activity timestamp.");
      return;
    }
    const activity = {
      user_id: form.userId.trim(),
      timestamp: localDate.toISOString(),
      login_time: form.loginTime || null,
      transaction_amount: amount,
      transaction_frequency: frequency,
      device_type: form.deviceType.trim() || null,
      device_id: form.deviceId.trim() || null,
      location: form.location.trim() || null,
      transaction_type: form.transactionType.trim() || null,
      activity_type: form.activityType.trim(),
      transaction_status: form.transactionStatus.trim() || null,
    };
    idempotencyKey.current ??= globalThis.crypto?.randomUUID?.() ?? null;
    await onAnalyze(
      { activity, recent_activities: history },
      idempotencyKey.current ?? undefined,
    );
  };

  return (
    <div className="page-stack">
      <div className="callout-note">
        <span className="callout-mark">i</span>
        <p>
          Submitted activity is analyzed only; it is <strong>not added</strong> to a trusted
          behavioral baseline. Prior events are optional inference context and are not verified
          by this dashboard.
        </p>
      </div>

      <div className="monitoring-grid">
        <Panel title="Analyze transaction / activity" subtitle="Fields are sent to the existing Flask API">
          <form className="analysis-form" onSubmit={submit}>
            <div className="form-grid">
              <FormField label="User ID" required>
                <input name="userId" value={form.userId} onChange={update} placeholder="e.g. user-001" required maxLength={128} />
              </FormField>
              <FormField label="Activity timestamp" required>
                <input type="datetime-local" name="timestamp" value={form.timestamp} onChange={update} required />
              </FormField>
              <FormField label="Login time">
                <input type="time" name="loginTime" value={form.loginTime} onChange={update} />
              </FormField>
              <FormField label="Transaction amount">
                <input type="number" min="0" step="0.01" name="transactionAmount" value={form.transactionAmount} onChange={update} placeholder="Optional" />
              </FormField>
              <FormField label="Transaction frequency">
                <input type="number" min="0" step="1" name="transactionFrequency" value={form.transactionFrequency} onChange={update} placeholder="Optional" />
              </FormField>
              <FormField label="Activity type" required>
                <input name="activityType" value={form.activityType} onChange={update} placeholder="login_and_transaction" required maxLength={64} />
              </FormField>
              <FormField label="Transaction type">
                <input name="transactionType" value={form.transactionType} onChange={update} placeholder="online" maxLength={64} />
              </FormField>
              <FormField label="Transaction status">
                <input name="transactionStatus" value={form.transactionStatus} onChange={update} placeholder="completed" maxLength={32} />
              </FormField>
              <FormField label="Device type">
                <input name="deviceType" value={form.deviceType} onChange={update} placeholder="mobile" maxLength={64} />
              </FormField>
              <FormField label="Device ID">
                <input name="deviceId" value={form.deviceId} onChange={update} placeholder="Optional" maxLength={128} />
              </FormField>
              <FormField label="Location">
                <input name="location" value={form.location} onChange={update} placeholder="Optional" maxLength={128} />
              </FormField>
            </div>
            <FormField label="Prior activities for sequence models" hint="Optional JSON array. Same user, chronological, all entries before this activity.">
              <textarea
                className="history-input"
                name="recentActivities"
                value={form.recentActivities}
                onChange={update}
                rows={4}
                spellCheck="false"
                aria-describedby="history-context-note"
              />
            </FormField>
            <p className="form-hint" id="history-context-note">
              Context is untrusted inference input and is never saved as verified history.
            </p>
            {historyError && <p className="inline-error" role="alert">{historyError}</p>}
            {submitError && <p className="inline-error" role="alert">{submitError}</p>}
            <div className="form-actions">
              <span className="form-footnote">Risk output is advisory and not a fraud confirmation.</span>
              <button className="button button-primary" type="submit" disabled={submitting}>
                {submitting ? <><span className="button-spinner" /> Analyzing</> : "Analyze activity"}
              </button>
            </div>
          </form>
        </Panel>

        <Panel
          title="Latest result"
          subtitle="Most recent API response in this browser session"
          action={selectedAnalysis && <RiskBadge category={selectedAnalysis.risk_assessment?.risk_category} />}
        >
          {selectedAnalysis ? (
            <div className="latest-result">
              <div className="result-score">
                <strong>{Number.isFinite(selectedAnalysis.risk_assessment?.risk_score) ? `${formatNumber(selectedAnalysis.risk_assessment.risk_score, 1)}` : "—"}</strong>
                <span>TRIAGE SCORE / 100</span>
              </div>
              <div className="result-meta">
                <ResultMeta label="User" value={selectedAnalysis.user_id} />
                <ResultMeta label="Amount" value={formatMoney(selectedAnalysis.transaction?.amount)} />
                <ResultMeta label="Recommended action" value={selectedAnalysis.recommended_action?.replaceAll("_", " ")} />
              </div>
              {selectedAnalysis.risk_assessment?.status === "unavailable" && (
                <div className="unavailable-note">
                  Risk score unavailable. No usable model output was available for this request.
                </div>
              )}
            </div>
          ) : (
            <EmptyState title="Awaiting an analysis" description="The latest response will appear here after submission." />
          )}
        </Panel>
      </div>

      <Panel
        title="Analyzed activities"
        subtitle={
          historyAvailable
            ? `Persistent MongoDB history · ${formatNumber(historyCount)} records · page ${historyPage} of ${Math.max(historyTotalPages, 1)}`
            : "Persistent transaction history from Flask"
        }
        action={
          <div className="table-controls">
            <input
              className="search-input"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search this page"
              aria-label="Search current transaction page"
            />
            <select value={riskFilter} onChange={(event) => setRiskFilter(event.target.value)} aria-label="Filter by risk category">
              <option value="ALL">All risk levels</option>
              <option value="LOW RISK">Low risk</option>
              <option value="MEDIUM RISK">Medium risk</option>
              <option value="HIGH RISK">High risk</option>
              <option value="UNAVAILABLE">Unavailable</option>
            </select>
          </div>
        }
      >
        {filteredAnalyses.length ? (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Activity time</th>
                  <th>User ID</th>
                  <th>Transaction</th>
                  <th>Amount</th>
                  <th>Risk</th>
                  <th>Score</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {filteredAnalyses.map((item) => (
                  <tr key={item.request_id}>
                    <td>{formatDate(item.transaction?.timestamp)}</td>
                    <td className="mono">{item.user_id}</td>
                    <td>{item.transaction?.type || "—"}</td>
                    <td>{formatMoney(item.transaction?.amount)}</td>
                    <td><RiskBadge category={item.risk_assessment?.risk_category} /></td>
                    <td>{Number.isFinite(item.risk_assessment?.risk_score) ? formatNumber(item.risk_assessment.risk_score, 1) : "—"}</td>
                    <td>
                      <button className="row-action" type="button" onClick={() => onOpenResult(item)} aria-label={`Review analysis for ${item.user_id}`}>
                        →
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            title={
              historyLoading
                ? "Loading transaction history"
                : !historyAvailable
                  ? "Persistent history unavailable"
                  : analyses.length
                    ? "No matching analyses"
                    : "No activities analyzed yet"
            }
            description={
              historyAvailable
                ? analyses.length
                  ? "Try changing the search or risk filter."
                  : "Submit an activity above to create a persistent analysis record."
                : "Check MongoDB connectivity and refresh the dashboard."
            }
          />
        )}
        {historyAvailable && (
          <div className="pagination-controls">
            <span>
              {formatNumber(historyCount)} total · page {historyPage} of {Math.max(historyTotalPages, 1)}
            </span>
            <div>
              <button
                className="button button-secondary"
                type="button"
                onClick={() => onHistoryPageChange((page) => Math.max(1, page - 1))}
                disabled={historyPage <= 1 || historyLoading}
              >
                Previous
              </button>
              <button
                className="button button-secondary"
                type="button"
                onClick={() => onHistoryPageChange((page) => Math.min(historyTotalPages, page + 1))}
                disabled={historyPage >= historyTotalPages || historyLoading}
              >
                Next
              </button>
            </div>
          </div>
        )}
      </Panel>
    </div>
  );
}

function BehaviorPage({ users, latestAnalysis }) {
  const [selectedUserId, setSelectedUserId] = useState("");
  const [profile, setProfile] = useState(null);
  const [loadingProfile, setLoadingProfile] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!selectedUserId && users.length) setSelectedUserId(users[0].user_id);
  }, [selectedUserId, users]);

  useEffect(() => {
    if (!selectedUserId) {
      setProfile(null);
      return undefined;
    }
    let active = true;
    setLoadingProfile(true);
    setError("");
    api.user(selectedUserId)
      .then((value) => {
        if (active) setProfile(value);
      })
      .catch((cause) => {
        if (active) setError(cause.message);
      })
      .finally(() => {
        if (active) setLoadingProfile(false);
      });
    return () => {
      active = false;
    };
  }, [selectedUserId]);

  const deviations =
    latestAnalysis?.user_id === selectedUserId
      ? latestAnalysis.behavioral?.deviations ?? []
      : [];

  return (
    <div className="page-stack">
      <Panel
        title="Trusted user profiles"
        subtitle="Only profiles loaded by the backend's trusted source are listed"
        action={
          <select
            className="profile-select"
            value={selectedUserId}
            onChange={(event) => setSelectedUserId(event.target.value)}
            aria-label="Select a user profile"
          >
            <option value="">Select a user</option>
            {users.map((user) => <option value={user.user_id} key={user.user_id}>{user.user_id}</option>)}
          </select>
        }
      >
        {error && <div className="inline-error" role="alert">{error}</div>}
        {loadingProfile ? (
          <LoadingState label="Loading profile" />
        ) : profile ? (
          <div className="profile-content">
            <div className="profile-summary">
              <div className="profile-user">
                <span className="avatar-mark">{profile.user_id?.slice(0, 1)?.toUpperCase() || "U"}</span>
                <div>
                  <strong className="mono">{profile.user_id}</strong>
                  <span>Trusted activity baseline</span>
                </div>
              </div>
              <StatusPill label={profile.state} tone={profile.state === "ESTABLISHED USER" ? "green" : "orange"} />
            </div>
            <div className="profile-metric-grid">
              <ProfileMetric label="Trusted activities" value={formatNumber(profile.trusted_activity_count)} />
              <ProfileMetric label="History span" value={`${formatNumber(profile.history_days, 1)} days`} />
              <ProfileMetric label="Average amount" value={formatMoney(profile.average_transaction_amount)} />
              <ProfileMetric label="Average frequency" value={formatNumber(profile.average_transaction_frequency, 1)} />
              <ProfileMetric label="Typical login hours" value={profile.typical_login_hours?.length ? profile.typical_login_hours.map((hour) => `${String(hour).padStart(2, "0")}:00`).join(", ") : "Unavailable"} />
              <ProfileMetric label="Known devices / locations" value={`${formatNumber(profile.known_device_count)} / ${formatNumber(profile.known_location_count)}`} />
            </div>
            <p className="privacy-note">
              Raw device identifiers, location values, and transaction histories are intentionally not exposed by the profile API.
            </p>
          </div>
        ) : (
          <EmptyState
            title={users.length ? "Select a profile" : "No trusted profiles available"}
            description={users.length ? "Choose a user to view their baseline summary." : "New and learning users are not added as trusted profiles automatically."}
          />
        )}
      </Panel>

      <Panel
        title="Behavioral deviations"
        subtitle={selectedUserId ? `Latest available analysis for ${selectedUserId}` : "Deviations are shown after selecting a user"}
      >
        {deviations.length ? (
          <div className="deviation-list">
            {deviations.map((deviation, index) => (
              <div className="deviation-row" key={`${deviation.code}-${index}`}>
                <span className="deviation-indicator" />
                <div>
                  <strong>{deviation.code?.replaceAll("_", " ") || "Context deviation"}</strong>
                  <p>{deviation.message || "A contextual difference was reported."}</p>
                  <div className="deviation-detail">
                    Observed: {String(deviation.observed ?? "—")} <span>·</span> Expected: {Array.isArray(deviation.expected) ? deviation.expected.join(", ") : typeof deviation.expected === "object" && deviation.expected ? JSON.stringify(deviation.expected) : String(deviation.expected ?? "—")}
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <EmptyState
            title="No deviations to display"
            description={latestAnalysis ? "The latest analysis for this profile contained no contextual deviations, or belongs to another user." : "Contextual comparisons appear after an activity is analyzed for an established user."}
          />
        )}
      </Panel>
    </div>
  );
}

function DetectionPage({ analysis }) {
  if (!analysis) {
    return <EmptyPage title="No detection output yet" description="Analyze an activity to inspect the separate model signals." />;
  }
  return (
    <div className="page-stack">
      <div className="callout-note">
        <span className="callout-mark">i</span>
        <p>Model and anomaly signals are advisory. A behavioral anomaly is not a confirmed fraud decision.</p>
      </div>
      <div className="signal-grid">
        <SignalCard
          title="Random Forest classifier"
          subtitle="Initial transaction signal · cold-start capable"
          result={analysis.classifier}
          value={analysis.classifier?.status === "available" ? formatNumber(analysis.classifier.fraud_probability * 100, 2) : null}
          unit={analysis.classifier?.status === "available" ? "model score · not calibrated likelihood" : null}
          detail={analysis.classifier?.status === "available" ? `${analysis.classifier.prediction} · threshold ${formatNumber(analysis.classifier.decision_threshold * 100, 2)}%` : analysis.classifier?.reason}
        />
        <SignalCard
          title="LSTM sequence analysis"
          subtitle="Sequential behavioral anomaly signal"
          result={analysis.lstm}
          value={analysis.lstm?.status === "available" ? formatNumber(analysis.lstm.anomaly_score * 100, 2) : null}
          unit={analysis.lstm?.status === "available" ? "anomaly score · not a fraud probability" : null}
          detail={analysis.lstm?.status === "available" ? `Threshold ${formatNumber(analysis.lstm.threshold * 100, 2)}% · ${analysis.lstm.behavioral_anomaly ? "unusual sequence flagged" : "below anomaly threshold"}` : "Requires an established user and sufficient chronological context."}
        />
        <SignalCard
          title="Autoencoder analysis"
          subtitle="Normal-activity reconstruction error"
          result={analysis.autoencoder}
          value={analysis.autoencoder?.status === "available" ? formatNumber(analysis.autoencoder.reconstruction_error, 4) : null}
          unit={analysis.autoencoder?.status === "available" ? "reconstruction error" : null}
          detail={analysis.autoencoder?.status === "available" ? `Validation threshold ${formatNumber(analysis.autoencoder.threshold, 4)} · ${analysis.autoencoder.behavioral_anomaly ? "threshold exceeded" : "within threshold"}` : "Requires an established user and sufficient chronological context."}
        />
      </div>
      <Panel title="Classifier decision" subtitle="The initial classifier result is reported separately from anomaly signals">
        {analysis.classifier?.status === "available" ? (
          <div className="decision-row">
            <RiskBadge category={analysis.classifier.risk_category} />
            <span className="decision-copy">
              Classifier output: <strong>{analysis.classifier.prediction}</strong>. This is a model output, not a confirmed outcome.
            </span>
          </div>
        ) : (
          <UnavailablePanel message={analysis.classifier?.reason || "No transaction classifier output was available."} />
        )}
      </Panel>
    </div>
  );
}

function RiskPage({ analysis }) {
  const assessment = analysis?.risk_assessment;
  if (!assessment) {
    return <EmptyPage title="No risk assessment yet" description="Submit an activity to receive a risk assessment from the Flask API." />;
  }
  if (assessment.status !== "available") {
    return (
      <div className="page-stack">
        <div className="unavailable-hero">
          <span className="unavailable-icon">—</span>
          <div>
            <p className="eyebrow">ASSESSMENT UNAVAILABLE</p>
            <h2>No combined score was produced</h2>
            <p>Missing model outputs are not treated as low-risk evidence. Follow the manual-review recommendation.</p>
          </div>
          <StatusPill label="Manual review" tone="orange" />
        </div>
        <Panel title="Assessment details" subtitle="Reasons reported by the backend">
          <ReasonList reasons={assessment.explanations} />
        </Panel>
      </div>
    );
  }
  const contributions = Object.entries(assessment.signals ?? {})
    .filter(([, signal]) => signal.available && Number.isFinite(signal.contribution_points))
    .map(([key, signal]) => ({
      name: signalTitle(key),
      points: signal.contribution_points,
      score: signal.normalized_score * 100,
      fill: signalColor(key),
    }));

  return (
    <div className="page-stack">
      <section className="risk-summary-card">
        <div className="risk-score-visual">
          <div className={`risk-score-ring ${riskClass(assessment.risk_category)}`}>
            <strong>{formatNumber(assessment.risk_score, 1)}</strong>
            <span>/ 100</span>
          </div>
        </div>
        <div className="risk-summary-copy">
          <p className="eyebrow">COMBINED TRIAGE ASSESSMENT</p>
          <h2><RiskBadge category={assessment.risk_category} /></h2>
          <p>{assessment.score_semantics}</p>
          <div className="risk-summary-meta">
            <span>{formatNumber(assessment.available_signal_count)} signals included</span>
            <span>·</span>
            <span>User state: {analysis.behavioral?.state ?? "Unknown"}</span>
          </div>
        </div>
        <div className={`recommendation-box ${assessment.requires_additional_verification ? "needs-review" : ""}`}>
          <span className="recommendation-label">RECOMMENDED ACTION</span>
          <strong>{assessment.recommendation?.replaceAll("_", " ") || "Continue monitoring"}</strong>
          <span>Fraud determination: not made</span>
        </div>
      </section>

      <div className="risk-detail-grid">
        <Panel title="Signal contributions" subtitle="Weighted contribution points; unavailable signals are excluded">
          {contributions.length ? (
            <div className="contribution-chart">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={contributions} layout="vertical" margin={{ top: 4, right: 12, left: 8, bottom: 4 }}>
                  <CartesianGrid stroke="#253247" horizontal={false} />
                  <XAxis type="number" domain={[0, "dataMax"]} tick={{ fill: "#8290a4", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis type="category" dataKey="name" width={142} tick={{ fill: "#b7c3d3", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip content={<ChartTooltip suffix=" points" />} />
                  <Bar dataKey="points" name="Contribution" radius={[0, 5, 5, 0]} barSize={17}>
                    {contributions.map((item) => <Cell key={item.name} fill={item.fill} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyState title="No weighted signals" description="No available signal contributed to this score." />
          )}
          <div className="contribution-list">
            {Object.entries(assessment.signals ?? {}).map(([key, signal]) => (
              <div className="contribution-item" key={key}>
                <span className={`contribution-swatch ${signal.available ? "" : "disabled"}`} style={signal.available ? { background: signalColor(key) } : undefined} />
                <div className="contribution-name">
                  <strong>{signalTitle(key)}</strong>
                  <span>{signal.available ? signal.normalization : signal.exclusion_reason || "Unavailable"}</span>
                </div>
                <div className="contribution-value">
                  {signal.available && Number.isFinite(signal.contribution_points)
                    ? `${formatNumber(signal.contribution_points, 2)} pts`
                    : "Unavailable"}
                </div>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Reasons & verification" subtitle="Explanations from available component signals">
          <ReasonList reasons={assessment.explanations} />
          {assessment.recommendation_reasons?.length > 0 && (
            <div className="recommendation-reasons">
              <h3>Verification guidance</h3>
              {assessment.recommendation_reasons.map((reason) => <p key={reason}>{reason}</p>)}
            </div>
          )}
          <div className="no-verdict-note">
            This system surfaces risk signals for review. It does not confirm that an activity is fraudulent.
          </div>
        </Panel>
      </div>
      {assessment.limitations?.length > 0 && (
        <Panel title="Model limitations" subtitle="Research context reported by the backend">
          <ul className="limitations-list">
            {assessment.limitations.map((item) => <li key={item}>{item}</li>)}
          </ul>
        </Panel>
      )}
    </div>
  );
}

function AlertsPage({
  alerts,
  count,
  page,
  totalPages,
  available,
  loading,
  onPageChange,
}) {
  return (
    <div className="page-stack">
      <div className="callout-note warning-callout">
        <span className="callout-mark">!</span>
        <p>Alerts indicate that additional verification was recommended. They are not confirmed fraud findings.</p>
      </div>
      <Panel
        title="Review queue"
        subtitle={
          available
            ? `${formatNumber(count)} persistent alerts · page ${page} of ${Math.max(totalPages, 1)}`
            : "Persistent alerts returned by the API"
        }
      >
        {alerts.length ? (
          <div className="alert-list">
            {alerts.map((alert) => <AlertRow alert={alert} key={alert.alert_id} />)}
          </div>
        ) : (
          <EmptyState
            title={
              loading
                ? "Loading review alerts"
                : available
                  ? "No review alerts"
                  : "Persistent alerts unavailable"
            }
            description={
              available
                ? "Alerts appear when the API recommends additional verification."
                : "Check MongoDB connectivity and refresh the dashboard."
            }
          />
        )}
        {available && (
          <div className="pagination-controls">
            <span>{formatNumber(count)} total · page {page} of {Math.max(totalPages, 1)}</span>
            <div>
              <button
                className="button button-secondary"
                type="button"
                onClick={() => onPageChange((current) => Math.max(1, current - 1))}
                disabled={page <= 1 || loading}
              >
                Previous
              </button>
              <button
                className="button button-secondary"
                type="button"
                onClick={() => onPageChange((current) => Math.min(totalPages, current + 1))}
                disabled={page >= totalPages || loading}
              >
                Next
              </button>
            </div>
          </div>
        )}
      </Panel>
    </div>
  );
}

function ReportsPage({ report, analyses, refreshing }) {
  const [downloadError, setDownloadError] = useState("");
  const counts = report?.risk_category_counts;
  const data = counts
    ? Object.entries(RISK_COLORS).map(([name, color]) => ({
        name,
        count: counts[name] ?? 0,
        color,
      }))
    : [];
  const amounts = analyses
    .map((item) => item.transaction?.amount)
    .filter(Number.isFinite);
  const sessionAverage = amounts.length
    ? amounts.reduce((sum, amount) => sum + amount, 0) / amounts.length
    : null;

  const download = () => {
    if (!report) {
      setDownloadError("No backend report is available to download.");
      return;
    }
    const output = {
      generated_at: new Date().toISOString(),
      scope: "API process report plus current browser session transaction summaries",
      backend_report: report,
      current_browser_session: analyses.map((item) => ({
        request_id: item.request_id,
        user_id: item.user_id,
        transaction: item.transaction,
        risk_category: item.risk_assessment?.risk_category ?? null,
        risk_score: item.risk_assessment?.risk_score ?? null,
        recommended_action: item.recommended_action,
      })),
    };
    const blob = new Blob([JSON.stringify(output, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `risk-report-${new Date().toISOString().slice(0, 10)}.json`;
    link.click();
    URL.revokeObjectURL(url);
    setDownloadError("");
  };

  return (
    <div className="page-stack">
      <Panel
        title="Assessment summary"
        subtitle={report?.storage || "Waiting for backend report data"}
        action={<button className="button button-secondary" type="button" onClick={download} disabled={!report}>Download JSON</button>}
      >
        {downloadError && <p className="inline-error" role="alert">{downloadError}</p>}
        <div className="report-metrics">
          <ProfileMetric label="Backend assessments" value={report ? formatNumber(report.assessments_total) : "—"} />
          <ProfileMetric label="Unavailable scores" value={report ? formatNumber(report.unavailable_assessments) : "—"} />
          <ProfileMetric label="Review alerts retained" value={report ? formatNumber(report.retained_alerts) : "—"} />
          <ProfileMetric label="Browser-session analyses" value={formatNumber(analyses.length)} />
          <ProfileMetric label="Session transactions with amount" value={formatNumber(amounts.length)} />
          <ProfileMetric label="Session average amount" value={formatMoney(sessionAverage)} />
        </div>
      </Panel>

      <Panel title="Risk distribution" subtitle="Persistent backend counts; no seeded or sample values">
        {data.some((item) => item.count > 0) ? (
          <div className="report-chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data} margin={{ top: 12, right: 12, left: 0, bottom: 4 }}>
                <CartesianGrid stroke="#253247" vertical={false} />
                <XAxis dataKey="name" tick={{ fill: "#a7b3c4", fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis allowDecimals={false} tick={{ fill: "#8290a4", fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip content={<ChartTooltip />} />
                <Bar dataKey="count" name="Assessments" radius={[5, 5, 0, 0]} barSize={42}>
                  {data.map((item) => <Cell key={item.name} fill={item.color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <EmptyState title={refreshing ? "Loading report" : "No risk distribution yet"} description="Charts appear after the backend records actual assessments." />
        )}
      </Panel>
      {report?.recent_assessments?.length > 0 && (
        <Panel title="Recent backend summaries" subtitle="The API returns minimized summaries without transaction or user details">
          <div className="table-scroll">
            <table>
              <thead><tr><th>Recorded at</th><th>Risk</th><th>Score</th><th>Signals</th><th>Recommended action</th></tr></thead>
              <tbody>
                {report.recent_assessments.slice(0, 20).map((item, index) => (
                  <tr key={`${item.created_at}-${index}`}>
                    <td>{formatDate(item.created_at)}</td>
                    <td><RiskBadge category={item.risk_category} /></td>
                    <td>{Number.isFinite(item.risk_score) ? formatNumber(item.risk_score, 1) : "—"}</td>
                    <td>{formatNumber(item.available_signal_count)}</td>
                    <td>{item.recommended_action?.replaceAll("_", " ") || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
    </div>
  );
}

function ModelAvailability({ models, database }) {
  const items = [
    ["classifier", "Random Forest classifier", "Cold-start transaction analysis"],
    ["lstm", "LSTM sequence model", "Sequential behavior signal"],
    ["autoencoder", "Autoencoder", "Reconstruction anomaly signal"],
  ];
  return (
    <div className="model-list">
      {items.map(([key, title, detail]) => {
        const model = models?.[key];
        return (
          <div className="model-row" key={key}>
            <span className={`model-icon ${model?.available ? "model-ready" : "model-missing"}`}>
              {model?.available ? "✓" : "—"}
            </span>
            <div className="model-copy">
              <strong>{title}</strong>
              <span>{detail}</span>
            </div>
            <StatusPill label={modelStatusLabel(model)} tone={model?.available ? "green" : "neutral"} />
          </div>
        );
      })}
      <div className="model-row">
        <span className={`model-icon ${database?.available ? "model-ready" : "model-missing"}`}>
          {database?.available ? "✓" : "—"}
        </span>
        <div className="model-copy">
          <strong>MongoDB persistence</strong>
          <span>{database?.status ?? "Database health unavailable"}</span>
        </div>
        <StatusPill
          label={database?.available ? "Connected" : "Unavailable"}
          tone={database?.available ? "green" : "neutral"}
        />
      </div>
    </div>
  );
}

function SignalCard({ title, subtitle, result, value, unit, detail }) {
  const available = result?.status === "available" && value !== null;
  return (
    <Panel
      title={title}
      subtitle={subtitle}
      className="signal-panel"
      action={<StatusPill label={available ? "Available" : signalStatusText(result)} tone={available ? result.behavioral_anomaly ? "orange" : "green" : "neutral"} />}
    >
      {available ? (
        <div className="signal-value">
          <strong>{value}</strong>
          <span>{unit}</span>
          <p>{detail}</p>
        </div>
      ) : (
        <UnavailablePanel message={detail || result?.reason || "This model output is not available for the current request."} />
      )}
    </Panel>
  );
}

function AlertRow({ alert, onClick }) {
  return (
    <div className="alert-row">
      <span className={`alert-severity ${riskClass(alert.risk_category)}`} />
      <div className="alert-main">
        <div className="alert-title-line">
          <strong>{alert.risk_category || "Review required"}</strong>
          <span className="alert-time">{formatDate(alert.created_at)}</span>
        </div>
        <p>{alert.reasons?.length ? alert.reasons.join(" ") : "Additional verification was recommended."}</p>
        <div className="alert-meta">
          <span className="mono">{alert.user_id || "User unavailable"}</span>
          <span>·</span>
          <span>{alert.recommended_action?.replaceAll("_", " ") || "manual review"}</span>
          <span>·</span>
          <span>Not a confirmed fraud finding</span>
        </div>
      </div>
      {onClick && (
        <button className="row-action" type="button" onClick={onClick} aria-label="Open alerts">
          →
        </button>
      )}
    </div>
  );
}

function MetricCard({ label, value, detail, glyph, tone }) {
  return (
    <div className="metric-card">
      <div className={`metric-glyph tone-${tone}`} aria-hidden="true">{glyph}</div>
      <div className="metric-copy">
        <span>{label}</span>
        <strong>{value}</strong>
        <small>{detail}</small>
      </div>
    </div>
  );
}

function ProfileMetric({ label, value }) {
  return (
    <div className="profile-metric">
      <span>{label}</span>
      <strong>{value ?? "—"}</strong>
    </div>
  );
}

function ResultMeta({ label, value }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value || "—"}</strong>
    </div>
  );
}

function FormField({ label, required, hint, children }) {
  return (
    <label className="form-field">
      <span className="field-label">{label}{required && <i aria-hidden="true"> *</i>}</span>
      {children}
      {hint && <small className="field-hint">{hint}</small>}
    </label>
  );
}

function Panel({ title, subtitle, action, className = "", children }) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-heading">
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {action && <div className="panel-action">{action}</div>}
      </div>
      {children}
    </section>
  );
}

function RiskBadge({ category }) {
  return (
    <span className={`risk-badge ${riskClass(category)}`}>
      <span className="risk-badge-dot" />
      {category || "UNAVAILABLE"}
    </span>
  );
}

function StatusPill({ label, tone = "neutral" }) {
  return <span className={`status-pill pill-${tone}`}>{label}</span>;
}

function EmptyState({ title, description, action }) {
  return (
    <div className="empty-state">
      <span className="empty-mark" aria-hidden="true">—</span>
      <strong>{title}</strong>
      <p>{description}</p>
      {action}
    </div>
  );
}

function EmptyPage({ title, description }) {
  return (
    <Panel title={title}>
      <EmptyState title={title} description={description} />
    </Panel>
  );
}

function LoadingState({ label }) {
  return <div className="loading-state"><span className="button-spinner" />{label}</div>;
}

function UnavailablePanel({ message }) {
  return (
    <div className="unavailable-note">
      <span className="unavailable-icon">—</span>
      <div>
        <strong>Signal unavailable</strong>
        <p>{message}</p>
      </div>
    </div>
  );
}

function ReasonList({ reasons }) {
  return reasons?.length ? (
    <ul className="reason-list">
      {reasons.map((reason, index) => <li key={`${reason}-${index}`}>{reason}</li>)}
    </ul>
  ) : (
    <EmptyState title="No explanations returned" description="No component explanation was available for this assessment." />
  );
}

function ChartTooltip({ active, payload, suffix = "" }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <span>{payload[0].name || payload[0].payload?.name}</span>
      <strong>{formatNumber(payload[0].value, 2)}{suffix}</strong>
    </div>
  );
}

function signalTitle(name) {
  return {
    initial_fraud_classifier: "Transaction classifier",
    lstm_behavior: "LSTM sequence",
    autoencoder: "Autoencoder",
    contextual_baseline: "Contextual baseline",
  }[name] || name;
}

function signalColor(name) {
  return {
    initial_fraud_classifier: "#35c6bd",
    lstm_behavior: "#8b7cf6",
    autoencoder: "#49a7ee",
    contextual_baseline: "#f4a340",
  }[name] || "#93a4ba";
}

function signalStatusText(result) {
  if (!result) return "No response";
  if (result.status === "not_active_for_user_state") return "Not active for user";
  if (result.status === "insufficient_context") return "Insufficient history";
  if (result.status === "not_configured") return "Not configured";
  if (result.status === "not_applicable") return "Not applicable";
  if (result.status === "unavailable") return "Unavailable";
  return result.status || "Unavailable";
}

function createInitialForm() {
  const now = new Date();
  const localNow = new Date(now.getTime() - now.getTimezoneOffset() * 60_000);
  return {
    userId: "",
    timestamp: localNow.toISOString().slice(0, 16),
    loginTime: now.toISOString().slice(11, 16),
    transactionAmount: "",
    transactionFrequency: "",
    deviceType: "",
    deviceId: "",
    location: "",
    transactionType: "",
    activityType: "",
    transactionStatus: "",
    recentActivities: "[]",
  };
}

export default App;
