# Empirical Design

The main sample is 2010–2018 applications. Core AI patents are defined by `G06N`.

Baseline application-level equation:

```text
Y_jkt = alpha + beta AI_j + X_j' gamma + IPC fixed effects + year fixed effects + error
```

Outcomes:

- grant probability
- grant delay
- claim reduction
- backward citations
- rejection-related citations
- inventor count
- applicant count

The field-year panel tests whether core AI patent density is associated with follow-on applications in the same IPC class.
