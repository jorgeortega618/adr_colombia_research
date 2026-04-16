# Contribution Matrix: Colombian ADRs vs S&P 500 and USD/COP

This matrix maps the scope of this research against prevalent empirical literature, establishing the novelty and robustness of the analytical framework. The focus is specifically on the Colombian equity market's unique dynamics—contrasting the energy sector against the banking sector—under dual macroeconomic exposure.

| Literature Benchmark / Prior Studies | Standard Approach | Our Novel Extension / Contribution | Rationale / Justification |
| :--- | :--- | :--- | :--- |
| **Emerging Market ADRs (General)** | Aggregate indices (e.g., MSCI EM) or generic Latin American bundles. | Strict focus on Colombian ADRs on the NYSE. | Colombia operates under a unique floating exchange regime highly sensitive to commodities, yet is often lost in aggregate LatAm studies. |
| **Sectoral Heterogeneity** | Broad equity index returns ignoring intra-market dispersion. | Segmenting analysis: Energy (Ecopetrol) vs. Financials (Grupo Aval, Bancolombia). | Directly captures the differential impact of terms-of-trade shocks. Oil price dynamics affect Ecopetrol differently than the credit channel affects banks. |
| **Systemic Risk Measurement** | Univariate Beta (CAPM) using only the S&P 500 or global index. | **Multivariate Conditional Beta:** $R_{i,t} = \alpha_t + \beta_t R_{SP500,t} + \gamma_t R_{USD/COP,t} + \epsilon_t$ | S&P exposure is measured *controlling* for FX volatility. Without this, beta estimates suffer from omitted variable bias due to EM currency depreciation masking equity risk. |
| **Time-Varying Dynamics** | Arbitrary sub-sampling (e.g., dividing data strictly pre/post COVID-19). | **Rolling OLS with HAC Errors:** Continuous 126, 252, and 504-day window tracking, validated with Newey-West standard errors. | Provides a non-parametric view of shifting risk premiums without assuming artificial/discrete sample breaks. |
| **Structural Breaks** | Exogenous assumptions (defining breaks a priori) or simple Bai-Perron. | **PELT Algorithm with Stability Criterion:** Endogenous detection filtered by a $\pm 30$ day consensus rule across multiple penalty parameters (`pen = 5, 10, 20`). | Eliminates "tuning artifact" critiques. A structural shift is only codified if it manifests consistently regardless of algorithmic sensitivity adjustments. |
| **Volatility Modeling** | Baseline GARCH(1,1) with Normal innovations. | **Bounded Grid Search + Distribution Iteration:** Optimization across $p,q \in \{1,2\}$, EGARCH, GJR-GARCH under normal, Student's *t*, and GED errors. | Specifically addresses and resolves residual ARCH effects (frequently ignored in EM energy stocks) and captures asymmetric shock persistence accurately. |

---
*Document produced as part of the Colombian ADR Research Robustness methodology.*
