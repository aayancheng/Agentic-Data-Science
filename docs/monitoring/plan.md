# Monitoring Plan (Production Readiness)

## Signals & Thresholds
- **Data Drift**: PSI / KS on key features; trigger > 0.2 PSI
- **Performance**: Fraud Recall@1% FPR drop > 10% triggers investigation
- **Calibration**: ECE > 0.03 triggers recalibration
- **Fairness**: Parity metrics outside tolerance bands -> escalation

## Collection & Storage
- Batch logs with summary statistics
- Monthly detailed evaluation on recent window

## Alerts & Escalation
- Email/pager to model owner & risk
- Ticket auto-created with run artifacts

## Remediation
- Shadow deploy challenger and backfill metrics
- Retrain with recent window if sustained drift
