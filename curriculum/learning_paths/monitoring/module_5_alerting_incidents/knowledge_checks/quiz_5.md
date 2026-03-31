# Module 5 Knowledge Check: Alerting & Incident Response

## Question 1: Alert Severity Classification

**Question:**
Classify each incident by severity (P0-P4) and justify:

A. LLM API returning 500 errors for all requests
B. Response latency p95 increased from 1s to 1.5s
C. Hallucination rate increased from 3% to 8%
D. Grafana dashboard showing incorrect cost data
E. Token cost 20% above budget projection

**Answer:**
```
A. P0 - Critical
   Justification: Complete service outage, all users affected
   Response: Immediate all-hands, rollback if needed

B. P3 - Low
   Justification: Degraded but functional, 50% increase but still within SLO
   Response: Investigate during business hours

C. P1 - High
   Justification: Quality degradation affecting user trust, but service functional
   Response: On-call investigation within 15 minutes

D. P4 - Info
   Justification: Monitoring issue, not affecting users
   Response: Fix in next sprint

E. P2 - Medium
   Justification: Cost concern but not critical, needs investigation
   Response: On-call review within 1 hour
```

---

## Question 2: Alert Fatigue

**Question:**
Your team receives 200 alerts per week. 80% are false positives. 
On-call engineers are burning out. What steps would you take?

**Answer:**
```
Immediate Actions (Week 1):
1. Audit all alert rules
   - Document purpose of each alert
   - Identify alerts that never triggered incidents

2. Increase thresholds
   - Raise warning thresholds by 20-50%
   - Add "for" duration (require sustained issues)

3. Consolidate alerts
   - Group related alerts into single notification
   - Implement alert deduplication

Medium-term (Month 1):
4. Implement alert tiers
   - Page-worthy: Only truly urgent issues
   - Ticket-worthy: Needs attention but not urgent
   - Log-worthy: Informational only

5. Add runbook links
   - Every alert must have actionable runbook
   - If no runbook, question if alert is needed

Long-term (Ongoing):
6. Weekly alert review
   - Review all pages from previous week
   - Remove or tune alerts that didn't lead to action

7. Track alert metrics
   - False positive rate
   - Time to acknowledge
   - Action rate (alerts leading to action)

Target: < 5 pages per person per week, > 80% action rate
```

---

## Question 3: Incident Escalation

**Question:**
You're on-call. At 2 AM, you receive an alert:
- Hallucination rate: 15% (threshold: 10%)
- Error rate: Normal
- Latency: Normal

What's your response process?

**Answer:**
```
1. ACKNOWLEDGE (within 5 minutes)
   - Acknowledge alert in PagerDuty
   - Post in incident Slack channel
   - "Investigating hallucination rate spike"

2. ASSESS (5-15 minutes)
   - Check dashboard: Is this isolated or widespread?
   - Check recent deploys: Any changes in last 24 hours?
   - Check user reports: Any complaints in support?
   - Check related metrics: Token usage, model changes?

3. DIAGNOSE (15-30 minutes)
   - Sample recent responses: Confirm hallucinations
   - Check specific topics: Is it domain-specific?
   - Check model version: Any automatic updates?
   - Check RAG sources: Any data issues?

4. DECIDE (30-45 minutes)
   If isolated/minor:
   - Continue monitoring
   - Plan investigation for morning
   
   If widespread/significant:
   - Escalate to backup on-call
   - Consider rollback if recent deploy
   - Prepare user communication if needed

5. DOCUMENT
   - Start incident timeline
   - Record all actions taken
   - Note key findings
```

---

## Question 4: Runbook Design

**Question:**
What sections should every incident runbook include?
Create a template for "High Hallucination Rate" incident.

**Answer:**
```
Runbook Template Sections:

1. Overview
   - What this alert means
   - Expected behavior
   - Business impact

2. Immediate Actions
   - First 5 minutes
   - Who to notify
   - Communication templates

3. Diagnosis Steps
   - Dashboard links
   - Commands to run
   - What to look for

4. Remediation Steps
   - Known fixes
   - Rollback procedures
   - Workarounds

5. Escalation
   - Who to contact
   - When to escalate
   - Contact information

6. Verification
   - How to confirm fix
   - Metrics to monitor
   - Success criteria

7. Post-Incident
   - What to document
   - Follow-up actions
   - Prevention measures

---

Example: High Hallucination Rate Runbook

1. Overview
   - Alert: Hallucination rate > 10%
   - Impact: Users receiving incorrect information
   - Severity: P1 (High)

2. Immediate Actions
   - Acknowledge alert
   - Post in #incidents channel
   - Check if recent deploy

3. Diagnosis
   - Dashboard: grafana.example.com/hallucination
   - Sample responses: python scripts/sample_responses.py
   - Check RAG sources: Verify knowledge base freshness

4. Remediation
   - If recent deploy: Rollback to previous version
   - If RAG issue: Refresh knowledge base
   - If model issue: Switch to fallback model

5. Verification
   - Monitor hallucination rate for 30 minutes
   - Sample 50 responses manually
   - Confirm rate < 8%

6. Post-Incident
   - Create incident report
   - Schedule post-mortem
   - Update runbook with learnings
```

---

## Question 5: MTTR Analysis

**Question:**
Your team's metrics:
- MTTA (Mean Time to Acknowledge): 15 minutes
- MTTR (Mean Time to Resolve): 4 hours
- Industry benchmark MTTR: 1 hour

What might be causing high MTTR? How would you improve?

**Answer:**
```
Potential Causes:

1. Slow Diagnosis
   - Insufficient monitoring/observability
   - Missing runbooks
   - Complex system architecture

2. Slow Remediation
   - Manual deployment processes
   - No automated rollback
   - Approval bottlenecks

3. Knowledge Gaps
   - Team unfamiliar with system
   - Tribal knowledge not documented
   - High engineer turnover

4. Process Issues
   - Too many approvers needed
   - Unclear escalation paths
   - Communication overhead

Improvement Plan:

Phase 1 (Immediate):
- Create/update runbooks for top 5 incident types
- Implement one-click rollback capability
- Add diagnostic commands to alerts

Phase 2 (30 days):
- Conduct incident response training
- Implement chatops for faster communication
- Add automated diagnostics to alerts

Phase 3 (90 days):
- Implement auto-remediation for common issues
- Improve monitoring coverage
- Regular game days/practice incidents

Target Metrics:
- MTTA: < 5 minutes
- MTTR: < 1 hour
- Auto-remediation rate: > 50%
```

---

*End of Module 5 Knowledge Check*
