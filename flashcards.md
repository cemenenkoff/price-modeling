<!-- omit in toc -->
# Price Modeling Flashcards

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [Q: What is elasticity of demand?](#q-what-is-elasticity-of-demand)
- [Q: What is price elasticity?](#q-what-is-price-elasticity)
- [Q: What are dynamic pricing algorithms?](#q-what-are-dynamic-pricing-algorithms)
- [Q: What are the most common techniques used in dynamic pricing algorithms?](#q-what-are-the-most-common-techniques-used-in-dynamic-pricing-algorithms)
- [Q: What are common techniques for measuring consumer preferences?](#q-what-are-common-techniques-for-measuring-consumer-preferences)
- [Q: What is conjoint analysis?](#q-what-is-conjoint-analysis)
- [Q: What are common methods in time series forecasting?](#q-what-are-common-methods-in-time-series-forecasting)
- [Q: What is SARIMAX modeling?](#q-what-is-sarimax-modeling)
- [Q: What is Prophet modeling?](#q-what-is-prophet-modeling)
- [Q: What is behavioral pricing?](#q-what-is-behavioral-pricing)
- [Q: What is reference pricing?](#q-what-is-reference-pricing)
- [Q: What is anchoring?](#q-what-is-anchoring)
- [Q: What is econometric modeling?](#q-what-is-econometric-modeling)
- [Q: What are generalized linear models?](#q-what-are-generalized-linear-models)
- [Q: What is logistic regression?](#q-what-is-logistic-regression)
- [Q: What is Poisson regression?](#q-what-is-poisson-regression)
- [Q: What is causal inference?](#q-what-is-causal-inference)
- [Q: What are the most common causal inference techniques?](#q-what-are-the-most-common-causal-inference-techniques)
- [Q: What is difference-in-differences (DiD)?](#q-what-is-difference-in-differences-did)
- [Q: What are instrumental variables (IV)?](#q-what-are-instrumental-variables-iv)
- [Q: What is propensity score matching (PSM)?](#q-what-is-propensity-score-matching-psm)
- [Q: What is maximum likelihood estimation (MLE)?](#q-what-is-maximum-likelihood-estimation-mle)
- [Q: What is the theory behind MLE?](#q-what-is-the-theory-behind-mle)
- [Q: How is MLE implemented?](#q-how-is-mle-implemented)
- [Q: What is hypothesis testing?](#q-what-is-hypothesis-testing)
- [Q: What is a p-value?](#q-what-is-a-p-value)
- [Q: What is a confidence interval?](#q-what-is-a-confidence-interval)
- [Q: What types of errors are there?](#q-what-types-of-errors-are-there)
- [Q: What is a Type I error?](#q-what-is-a-type-i-error)
- [Q: What is a Type II error?](#q-what-is-a-type-ii-error)
- [Q: How can Type I and Type II Errors be Summarized?](#q-how-can-type-i-and-type-ii-errors-be-summarized)
- [Q: What are Bayesian methods?](#q-what-are-bayesian-methods)
- [Q: What is Bayesian regression?](#q-what-is-bayesian-regression)
- [Q: What is hierarchical modeling?](#q-what-is-hierarchical-modeling)
- [Q: What is Design of Experiments (DOE)?](#q-what-is-design-of-experiments-doe)
- [Q: What are the core principles of DOE?](#q-what-are-the-core-principles-of-doe)
- [Q: What is A/B Testing?](#q-what-is-ab-testing)
- [Q: What are multi-armed bandits?](#q-what-are-multi-armed-bandits)
- [Q: Explain the exploration-exploitation tradeoff for adaptive environments.](#q-explain-the-exploration-exploitation-tradeoff-for-adaptive-environments)
- [Q: What are A/B testing metrics?](#q-what-are-ab-testing-metrics)
- [Q: What is statistical power?](#q-what-is-statistical-power)
- [Q: What are tree-based machine learning methods?](#q-what-are-tree-based-machine-learning-methods)
- [Q: What are gradient-boosted decision trees (GBDT)?](#q-what-are-gradient-boosted-decision-trees-gbdt)
- [Q: What are random forests?](#q-what-are-random-forests)
- [Q: What is XGBoost?](#q-what-is-xgboost)
- [Q: Explain L1/L2 regularization.](#q-explain-l1l2-regularization)
- [Q: What is ridge regression?](#q-what-is-ridge-regression)
- [Q: What is lasso regression?](#q-what-is-lasso-regression)
- [Q: What is a neural network (NN)?](#q-what-is-a-neural-network-nn)
- [Q: What is a convolutional neural network (CNN)?](#q-what-is-a-convolutional-neural-network-cnn)
- [Q: What is a recurrent neural network (RNN)?](#q-what-is-a-recurrent-neural-network-rnn)
- [Q: Explain the similarities between CNNs and RNNs.](#q-explain-the-similarities-between-cnns-and-rnns)
- [Q: Explain the differences between CNNs and RNNs.](#q-explain-the-differences-between-cnns-and-rnns)
- [Q: What are embeddings in the context of NNs?](#q-what-are-embeddings-in-the-context-of-nns)
- [Q: What is sequence modeling for demand prediction?](#q-what-is-sequence-modeling-for-demand-prediction)
- [Q: What is the concept of dimensionality reduction?](#q-what-is-the-concept-of-dimensionality-reduction)
- [Q: What is Principal Component Analysis (PCA)?](#q-what-is-principal-component-analysis-pca)
- [Q: What is t-SNE?](#q-what-is-t-sne)
- [Q: What are some common feature engineering methods?](#q-what-are-some-common-feature-engineering-methods)
- [Q: What is CatBoost?](#q-what-is-catboost)
- [Q: What is linear programming?](#q-what-is-linear-programming)
- [Q: What are simplex algorithms?](#q-what-are-simplex-algorithms)
- [Q: What is convex optimization?](#q-what-is-convex-optimization)
- [Q: What are gradient-based methods?](#q-what-are-gradient-based-methods)
- [Q: What is stochastic gradient descent (SGD)?](#q-what-is-stochastic-gradient-descent-sgd)
- [Q: What is the Adam optimizer?](#q-what-is-the-adam-optimizer)
- [Q: What is the Agile methodology?](#q-what-is-the-agile-methodology)
- [Q: What is Scrum?](#q-what-is-scrum)
- [Q: What is Kanban?](#q-what-is-kanban)
- [Q: What is sprint planning?](#q-what-is-sprint-planning)
- [Q: Using an analogy of trees in a forest, describe Bayesian statistics.](#q-using-an-analogy-of-trees-in-a-forest-describe-bayesian-statistics)

<!-- TOC end -->

---

<!-- TOC --><a name="q-what-is-elasticity-of-demand"></a>
### Q: What is elasticity of demand?

**A:** Elasticity of demand measures how the quantity demanded of a good or service changes in response to a change in its price. It is expressed as the percentage change in quantity demanded divided by the percentage change in price. High elasticity indicates demand is sensitive to price changes, while low elasticity indicates demand is relatively insensitive. For example, the price of gasoline is relatively inelastic (in the short run).

---

<!-- TOC --><a name="q-what-is-price-elasticity"></a>
### Q: What is price elasticity?

**A:** Price elasticity refers to the degree to which the quantity demanded or supplied of a good changes in response to a price change. It is quantified as Price Elasticity of Demand (PED):

```math
\text{PED} = \frac{\% \Delta \text{Quantity Demanded}}{\% \Delta \text{Price}}
```

Values:
- **Elastic (PED > 1):** Demand changes significantly with price changes.
- **Inelastic (PED < 1):** Demand changes minimally with price changes.
- **Unitary (PED = 1):** Proportional change in demand and price.

---

<!-- TOC --><a name="q-what-are-dynamic-pricing-algorithms"></a>
### Q: What are dynamic pricing algorithms?

**A:** Dynamic pricing algorithms are computational models used to adjust prices in real time based on various factors such as demand, supply, competition, and customer behavior. These algorithms aim to optimize revenue and maximize profit by dynamically setting prices according to changing market conditions.

---

<!-- TOC --><a name="q-what-are-the-most-common-techniques-used-in-dynamic-pricing-algorithms"></a>
### Q: What are the most common techniques used in dynamic pricing algorithms?

**A:** The most common techniques used in dynamic pricing algorithms include:

1. **Rule-Based Pricing:** Uses predefined rules (e.g., if demand increases, raise the price).
2. **Time-Based Pricing:** Prices change based on time windows (e.g., airline tickets).
3. **Demand-Based Pricing:** Adjusts prices in response to real-time demand changes.
4. **Competitive Pricing:** Prices are dynamically adjusted based on competitors’ pricing.
5. **Machine Learning Models:** Algorithms learn from historical data to predict optimal prices.
6. **Inventory-Based Pricing:** Prices depend on stock levels (e.g., lower inventory raises prices).
7. **Segmentation-Based Pricing:** Different prices for different customer segments.

---

<!-- TOC --><a name="q-what-are-common-techniques-for-measuring-consumer-preferences"></a>
### Q: What are common techniques for measuring consumer preferences?

**A:** Common techniques for measuring consumer preferences include:

1. **Surveys and Questionnaires:** Directly ask consumers about their preferences.
2. **Conjoint Analysis:** Measures trade-offs consumers make between different product features.
3. **A/B Testing:** Compare different versions of a product or service to see which is preferred.
4. **Choice Modeling:** Analyzes choices consumers make to infer preferences.
5. **Focus Groups:** Small groups discuss products to provide qualitative insights.
6. **Purchase Data Analysis:** Examines historical purchase behavior to understand preferences.
7. **Eye-Tracking Studies:** Measure where consumers focus their attention.

---

<!-- TOC --><a name="q-what-is-conjoint-analysis"></a>
### Q: What is conjoint analysis?

**A:** Conjoint analysis is a statistical technique used to determine how consumers value different features of a product or service. By presenting consumers with different combinations of features and asking them to choose their preferred option, conjoint analysis helps identify the trade-offs they are willing to make. This technique is commonly used in product design, pricing strategies, and market research.

---

<!-- TOC --><a name="q-what-are-common-methods-in-time-series-forecasting"></a>
### Q: What are common methods in time series forecasting?

**A:** Common methods in time series forecasting include:

1. **Naive Forecasting:** Uses the most recent observation as the forecast for future periods.
2. **Moving Averages (MA):** Averages the last *n* observations to smooth fluctuations.
3. **Exponential Smoothing (ETS):** Gives more weight to recent observations using a smoothing parameter.
4. **Autoregressive Integrated Moving Average (ARIMA):** Combines autoregression, differencing, and moving averages to model time series data.
5. **Seasonal ARIMA (SARIMA):** Extends ARIMA to include seasonality.
6. **SARIMAX:** SARIMA with exogenous variables to include external factors.
7. **Prophet:** A model developed by Facebook for handling seasonality and missing data.
8. **Long Short-Term Memory (LSTM):** A type of neural network for capturing long-term dependencies.
9. **Vector Autoregression (VAR):** For multivariate time series data with interdependent variables.
10. **GARCH (Generalized Autoregressive Conditional Heteroskedasticity):** Models volatility in financial time series.

---

<!-- TOC --><a name="q-what-is-sarimax-modeling"></a>
### Q: What is SARIMAX modeling?

**A:** SARIMAX (Seasonal Autoregressive Integrated Moving Average with eXogenous regressors) is an extension of SARIMA that incorporates external predictors (exogenous variables). It models time series data by considering:

- **Seasonality:** Accounts for recurring patterns (e.g., monthly, quarterly).
- **Autoregression (AR):** Uses past values to predict future values.
- **Differencing (I):** Removes trends by subtracting prior observations.
- **Moving Average (MA):** Accounts for past forecast errors.
- **Exogenous Variables (X):** External factors that influence the time series.

SARIMAX is useful when external factors (e.g., promotions, holidays) impact the target variable.

---

<!-- TOC --><a name="q-what-is-prophet-modeling"></a>
### Q: What is Prophet modeling?

**A:** Prophet is a forecasting tool developed by Facebook designed for handling time series data with daily, weekly, and yearly seasonality. It works well with missing data, outliers, and changing trends. Key features include:

- **Additive Model:** Combines trend, seasonality, and holiday effects.
- **Automatic Seasonality Detection:** Handles multiple seasonalities.
- **Flexible Trend Adjustment:** Allows for trend changes or "changepoints."
- **User-Friendly:** Minimal tuning required for accurate forecasts.

Prophet is particularly suited for business applications and data with strong seasonal patterns.

---

<!-- TOC --><a name="q-what-is-behavioral-pricing"></a>
### Q: What is behavioral pricing?

**A:** Behavioral pricing is a strategy that incorporates insights from psychology and behavioral economics to understand how consumers perceive prices and make purchasing decisions. It leverages cognitive biases, heuristics, and emotional responses to influence consumer behavior. Techniques in behavioral pricing include:

- **Charm Pricing:** Setting prices just below round numbers (e.g., $9.99 instead of $10.00).
- **Bundling:** Offering products together to influence perceived value.
- **Decoy Pricing:** Presenting a less attractive option to make another option seem more appealing.
- **Framing:** Presenting prices in a way that emphasizes gains or discounts.

---

<!-- TOC --><a name="q-what-is-reference-pricing"></a>
### Q: What is reference pricing?

**A:** Reference pricing is the strategy of setting a price based on consumers’ expectations or perceptions of what a fair price should be. These expectations are influenced by:

- **Past Prices:** Previous prices of the same product.
- **Competitor Prices:** Prices of similar products in the market.
- **Suggested Retail Prices:** Manufacturer’s recommended price.
- **Internal Benchmarks:** Consumers’ personal valuation of the product.

Reference pricing is often used to highlight discounts or make products appear more affordable by comparing them to a higher “reference” price.

---

<!-- TOC --><a name="q-what-is-anchoring"></a>
### Q: What is anchoring?

**A:** Anchoring is a cognitive bias where individuals rely heavily on the first piece of information (the "anchor") when making decisions. In pricing, the initial price shown to consumers influences their perception of subsequent prices. Examples of anchoring include:

- **Original vs. Discounted Prices:** Showing a high original price to make a discount seem more attractive (e.g., “Was $200, Now $150”).
- **High Initial Offers:** Starting with a high price to make later price reductions seem reasonable.
- **Price Comparisons:** Displaying premium options alongside standard options to make the standard option appear more affordable.

Anchoring helps shape consumers' willingness to pay based on the initial price they encounter.

---

<!-- TOC --><a name="q-what-is-econometric-modeling"></a>
### Q: What is econometric modeling?

**A:** Econometric modeling uses statistical techniques to quantify and analyze economic relationships based on empirical data. These models help test hypotheses, forecast trends, and understand causal relationships. Key components of econometric modeling include:

- **Data Collection:** Gathering economic or business data.
- **Model Specification:** Defining the mathematical relationship between variables.
- **Estimation:** Using methods like Ordinary Least Squares (OLS) to estimate parameters.
- **Hypothesis Testing:** Evaluating the significance of relationships.
- **Forecasting:** Predicting future outcomes based on model findings.

Examples of econometric models include **linear regression, time series models (ARIMA), and panel data models**.

---

<!-- TOC --><a name="q-what-are-generalized-linear-models"></a>
### Q: What are generalized linear models?

**A:** Generalized Linear Models (GLMs) are an extension of linear regression models that allow for response variables with different types of distributions (not just normal). GLMs consist of three main components:

1. **Random Component:** The distribution of the response variable (e.g., normal, binomial, Poisson).
2. **Systematic Component:** A linear combination of predictor variables.
3. **Link Function:** A function that links the linear predictor to the mean of the response variable (e.g., logit, log, identity).

**Common types of GLMs:**
- **Linear Regression:** For continuous outcomes (normal distribution).
- **Logistic Regression:** For binary outcomes (binomial distribution).
- **Poisson Regression:** For count data (Poisson distribution).

---

<!-- TOC --><a name="q-what-is-logistic-regression"></a>
### Q: What is logistic regression?

**A:** Logistic regression is a type of Generalized Linear Model (GLM) used for predicting binary outcomes (e.g., yes/no, 0/1). Instead of fitting a straight line, logistic regression models the probability of an event occurring using the **logit function**:

$$
\text{logit}(p) = \ln\left(\frac{p}{1 - p}\right) = \beta_0 + \beta_1x_1 + \ldots + \beta_nx_n
$$

Where:
- $p$ is the probability of the positive outcome.
- $\beta$ values are the coefficients estimated during training.

**Key Features:**
- **Output:** Probabilities between 0 and 1.
- **Decision Boundary:** Often used with a threshold (e.g., $p > 0.5$).
- **Applications:** Classification problems such as fraud detection, medical diagnosis, and customer churn prediction.

---

<!-- TOC --><a name="q-what-is-poisson-regression"></a>
### Q: What is Poisson regression?

**A:** Poisson regression is a type of Generalized Linear Model (GLM) used for modeling count data, where the response variable represents the number of times an event occurs. It assumes that the response variable follows a **Poisson distribution** and is typically used when the data are non-negative integers (e.g., 0, 1, 2, ...). The model uses a **log link function** to relate the linear predictor to the mean of the response variable:

$$
\log(\lambda) = \beta_0 + \beta_1 x_1 + \ldots + \beta_n x_n
$$

Where:
- $\lambda$ is the expected count (mean) of the response variable.
- $\beta$ values are the coefficients estimated during training.

**Key Characteristics:**
- Suitable for **count data** (e.g., number of customer purchases, number of website clicks).
- Assumes the **mean and variance are equal** (a property of the Poisson distribution).
- Can be extended to **Negative Binomial Regression** if the variance exceeds the mean (overdispersion).

**Applications:**
- Predicting the number of calls to a call center.
- Modeling the number of accidents at a traffic intersection.
- Analyzing the frequency of disease occurrences in epidemiology.

---

<!-- TOC --><a name="q-what-is-causal-inference"></a>
### Q: What is causal inference?

**A:** Causal inference is the process of determining whether a cause-and-effect relationship exists between variables. Unlike correlation, which only identifies associations, causal inference aims to understand how changes in one variable (the **treatment**) impact another variable (the **outcome**). This process often involves controlling for confounding factors and using statistical methods to isolate the causal effect.

Causal inference is essential in fields like economics, healthcare, and social sciences for making policy decisions and evaluating interventions.

---

<!-- TOC --><a name="q-what-are-the-most-common-causal-inference-techniques"></a>
### Q: What are the most common causal inference techniques?

**A:** The most common causal inference techniques include:

1. **Randomized Controlled Trials (RCTs):** Participants are randomly assigned to treatment and control groups to eliminate bias.
2. **Difference-in-Differences (DiD):** Compares the changes in outcomes over time between a treatment group and a control group.
3. **Instrumental Variables (IV):** Uses an external variable (instrument) to address endogeneity and estimate causal effects.
4. **Regression Discontinuity Design (RDD):** Exploits a cutoff or threshold to identify causal effects.
5. **Propensity Score Matching (PSM):** Matches treatment and control units with similar characteristics to control for confounding.
6. **Matching Methods:** Pairing units with similar covariates to isolate the treatment effect.

These methods help address biases and confounding factors to estimate causal relationships accurately.

---

<!-- TOC --><a name="q-what-is-difference-in-differences-did"></a>
### Q: What is difference-in-differences (DiD)?

**A:** **Difference-in-Differences (DiD)** is a causal inference technique used to estimate the effect of a treatment or intervention by comparing the changes in outcomes between a treatment group and a control group over time. DiD assumes that both groups would follow a **parallel trend** in the absence of the treatment.

**Steps:**
1. Measure the outcome for both groups before and after the intervention.
2. Calculate the difference in outcomes over time for each group.
3. Subtract the difference in the control group from the difference in the treatment group to estimate the treatment effect.

**Example:** Evaluating the impact of a new law on employment by comparing employment rates before and after the law in a state that adopted it versus a state that did not.

---

<!-- TOC --><a name="q-what-are-instrumental-variables-iv"></a>
### Q: What are instrumental variables (IV)?

**A:** **Instrumental Variables (IV)** is a method used in causal inference to address **endogeneity**—when an independent variable is correlated with the error term, leading to biased estimates. An instrumental variable helps isolate the causal effect by providing a source of variation that is correlated with the treatment but not with the error term.

**Conditions for a Valid Instrument:**
1. **Relevance:** The instrument is correlated with the treatment.
2. **Exogeneity:** The instrument is not correlated with the error term.

**Example:** Using the distance to the nearest college as an instrument to study the causal effect of education on earnings, assuming distance affects education levels but not earnings directly.

---

<!-- TOC --><a name="q-what-is-propensity-score-matching-psm"></a>
### Q: What is propensity score matching (PSM)?

**A:** **Propensity Score Matching (PSM)** is a technique used to estimate causal effects by matching treated and control units with similar characteristics (covariates). The propensity score is the probability of receiving the treatment, given observed characteristics.

**Steps:**
1. Estimate the propensity score using a logistic regression model.
2. Match each treated unit with one or more control units with similar propensity scores.
3. Compare outcomes between matched pairs to estimate the treatment effect.

**Advantages:**
- Reduces bias from confounding variables.
- Mimics the conditions of a randomized experiment.

**Applications:** Evaluating the effect of a new marketing campaign by matching customers who received the campaign with those who did not but have similar demographics.

---

<!-- TOC --><a name="q-what-is-maximum-likelihood-estimation-mle"></a>
### Q: What is maximum likelihood estimation (MLE)?

**A:** **Maximum Likelihood Estimation (MLE)** is a statistical method for estimating the parameters of a model by maximizing the likelihood function. The likelihood function measures how well the model explains the observed data for different parameter values. MLE finds the parameter estimates that make the observed data **most probable** under the assumed model.

For a given set of data points $X = \{x_1, x_2, \ldots, x_n\}$ and a parameter $ \theta $, MLE finds:

```math
\hat{\theta} = \arg\max_\theta \mathcal{L}(\theta | X)
```

Where $\mathcal{L}(\theta | X)$ is the likelihood function.

---

<!-- TOC --><a name="q-what-is-the-theory-behind-mle"></a>
### Q: What is the theory behind MLE?

**A:** The theory behind MLE is grounded in the principle of **likelihood**. Given a statistical model and observed data, the likelihood function represents the probability of observing the data for different parameter values. MLE selects the parameters that **maximize this likelihood**.

Key theoretical properties of MLE:

1. **Consistency:** As the sample size increases, the MLE converges to the true parameter value.
2. **Asymptotic Normality:** For large samples, the distribution of the MLE is approximately normal.
3. **Efficiency:** MLE achieves the lowest possible variance (the Cramér-Rao lower bound) among unbiased estimators for large samples.

The goal of MLE is to find the parameters that best explain the data by maximizing the likelihood of the observed outcomes.

---

<!-- TOC --><a name="q-how-is-mle-implemented"></a>
### Q: How is MLE implemented?

**A:** MLE is implemented using the following steps:

1. **Define the Likelihood Function:**
   Based on the assumed probability distribution of the data, define the likelihood function $\mathcal{L}(\theta | X)$. For independent observations, the likelihood is the product of individual probabilities:

```math
\mathcal{L}(\theta | X) = \prod_{i=1}^n f(x_i; \theta)
```

2. **Log-Likelihood Transformation:**
   To simplify the computation, take the natural log of the likelihood function (log-likelihood):

```math
\log \mathcal{L}(\theta | X) = \sum_{i=1}^n \log f(x_i; \theta)
```

3. **Maximize the Log-Likelihood:**
   Find the parameter \( \theta \) that maximizes the log-likelihood using calculus. This involves solving:

```math
\frac{d}{d\theta} \log \mathcal{L}(\theta | X) = 0
```

4. **Numerical Optimization:**
   For complex models, use numerical methods like **Gradient Descent**, **Newton-Raphson**, or **Expectation-Maximization (EM)** to find the maximum.

**Example in Python:**

```python
import numpy as np
from scipy.optimize import minimize

# Example: Estimating the mean of a normal distribution
def negative_log_likelihood(theta, data):
    mu, sigma = theta[0], theta[1]
    return -np.sum(np.log(1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((data - mu) / sigma)**2)))

data = np.array([2.3, 2.7, 3.1, 2.8, 3.0])
initial_guess = [0, 1]
result = minimize(negative_log_likelihood, initial_guess, args=(data,))
print("MLE estimates:", result.x)
```

---

<!-- TOC --><a name="q-what-is-hypothesis-testing"></a>
### Q: What is hypothesis testing?

**A:** **Hypothesis testing** is a statistical method used to make inferences about a population based on sample data. It involves evaluating whether there is enough evidence to reject a **null hypothesis ($H_0$)** in favor of an **alternative hypothesis ($H_1$)**.

**Steps in hypothesis testing:**

1. **Formulate Hypotheses:**
   - **Null Hypothesis ($H_0$)**: The assumption that there is no effect or difference (e.g., “The new drug has no effect”).
   - **Alternative Hypothesis ($H_1$)**: The claim being tested (e.g., “The new drug is effective”).

2. **Choose a Significance Level ($\alpha$)**: Commonly set at 0.05 (5%).

3. **Compute a Test Statistic**: Based on the sample data (e.g., z-score, t-score).

4. **Calculate the p-Value**: The probability of observing the data if the null hypothesis is true.

5. **Make a Decision**:
   - **Reject $H_0$** if $p \leq \alpha$.
   - **Fail to Reject $H_0$** if $p > \alpha$.

**Example:** Testing whether a new marketing strategy increases sales.

---

<!-- TOC --><a name="q-what-is-a-p-value"></a>
### Q: What is a p-value?

**A:** A **p-value** is the probability of obtaining a test statistic as extreme or more extreme than the observed value, assuming the null hypothesis ($H_0$) is true. It measures the strength of evidence against $H_0$.

**Interpretation:**
- **Small p-value ($p \leq \alpha$)**: Strong evidence against $H_0$; reject the null hypothesis.
- **Large p-value ($p > \alpha$)**: Insufficient evidence against $H_0$; fail to reject the null hypothesis.

**Example:**
A p-value of 0.03 means there is a 3% chance of observing the data if $H_0$ is true. If $\alpha = 0.05$, you would reject $H_0$.

---

<!-- TOC --><a name="q-what-is-a-confidence-interval"></a>
### Q: What is a confidence interval?

**A:** A **confidence interval (CI)** is a range of values that is likely to contain the true population parameter (e.g., mean or proportion) with a specified level of confidence (e.g., 95%).

**General Formula:**
$$
\text{CI} = \hat{\theta} \pm z_{\alpha/2} \cdot \text{SE}(\hat{\theta})
$$

Where:
- $\hat{\theta}$ = Sample estimate (e.g., sample mean).
- $z_{\alpha/2}$ = Critical value (e.g., 1.96 for 95% confidence).
- $\text{SE}(\hat{\theta})$ = Standard error of the estimate.

**Interpretation:**
A 95% confidence interval means that if the study were repeated many times, 95% of the intervals would contain the true parameter.

**Example:**
A 95% confidence interval for the average height might be (165 cm, 175 cm). This means you are 95% confident the true average height is between 165 cm and 175 cm.

---

<!-- TOC --><a name="q-what-types-of-errors-are-there"></a>
### Q: What types of errors are there?

**A:** In hypothesis testing, there are two primary types of errors:

1. **Type I Error (False Positive):** Rejecting the null hypothesis ($H_0$) when it is actually true.
2. **Type II Error (False Negative):** Failing to reject the null hypothesis ($H_0$) when it is actually false.

These errors are controlled by:

- **Significance Level ($\alpha$)**: Probability of a Type I error (commonly set at 0.05).
- **Power (1 - $\beta$)**: Probability of correctly rejecting $H_0$ (related to Type II error).

---

<!-- TOC --><a name="q-what-is-a-type-i-error"></a>
### Q: What is a Type I error?

**A:** A **Type I error** occurs when the null hypothesis ($H_0$) is **incorrectly rejected**, even though it is actually true. It is also called a **false positive**.

- **Probability of Type I Error**: $\alpha$ (significance level), commonly set at 0.05.

**Example:**
A medical test concludes a patient has a disease (positive result) when they do not actually have it.

---

<!-- TOC --><a name="q-what-is-a-type-ii-error"></a>
### Q: What is a Type II error?

**A:** A **Type II error** occurs when the null hypothesis ($H_0$) is **incorrectly accepted** (i.e., failing to reject it), even though it is actually false. It is also called a **false negative**.

- **Probability of Type II Error**: $\beta$.
- **Power**: $1 - \beta$, the probability of correctly rejecting $H_0$.

**Example:**
A medical test concludes a patient does not have a disease (negative result) when they actually do have it.

---

<!-- TOC --><a name="q-how-can-type-i-and-type-ii-errors-be-summarized"></a>
### Q: How can Type I and Type II Errors be Summarized?
**A:** **Using a decision matrix** is a great way to display the types of conclusions made from hypothesis testing.

| **Decision**                 | **$H_0$ True**          | **$H_0$ False**         |
|-------------------------------|---------------------------|---------------------------|
| **Reject $H_0$**           | Type I Error (False +)    | Correct Decision          |
| **Fail to Reject $H_0$**   | Correct Decision          | Type II Error (False -)   |

---

<!-- TOC --><a name="q-what-are-bayesian-methods"></a>
### Q: What are Bayesian methods?

**A:** **Bayesian methods** are statistical techniques based on **Bayes' Theorem**, which provides a way to update the probability of a hypothesis as new evidence or data becomes available. Bayesian methods incorporate prior beliefs and observed data to produce a **posterior distribution**.

**Bayes’ Theorem Formula:**

$$
P(\theta | X) = \frac{P(X | \theta) P(\theta)}{P(X)}
$$

Where:
- $P(\theta)$ Prior probability (belief before seeing the data).
- $P(X | \theta)$ Likelihood of the data given the parameter $\theta\).
- $P(X)$ Marginal likelihood (normalizing constant).
- $P(\theta | X)$ Posterior probability (updated belief).

**Applications:**
- Spam detection, medical diagnosis, risk assessment, machine learning.

---

<!-- TOC --><a name="q-what-is-bayesian-regression"></a>
### Q: What is Bayesian regression?

**A:** **Bayesian regression** is a regression technique that uses Bayesian inference to estimate the parameters of the model. Unlike traditional regression, Bayesian regression produces a **distribution of parameter estimates** (posterior distributions) instead of point estimates.

**Steps in Bayesian Regression:**

1. **Define a Prior Distribution** for the model parameters (e.g., normal distribution).
2. **Specify the Likelihood** based on the observed data.
3. **Compute the Posterior Distribution** using Bayes' Theorem.
4. **Draw Inferences** from the posterior (e.g., mean, credible intervals).

**Advantages:**
- Incorporates prior knowledge.
- Provides uncertainty estimates for parameters.
- Robust to small sample sizes.

**Example:** Predicting housing prices with uncertainty estimates for each coefficient.

---

<!-- TOC --><a name="q-what-is-hierarchical-modeling"></a>
### Q: What is hierarchical modeling?

**A:** **Hierarchical modeling** (also called **multilevel modeling**) is a statistical approach where data are structured in multiple levels or groups. It allows for the modeling of both **group-level** and **individual-level** effects simultaneously.

In hierarchical models:

- Parameters can vary **within groups** (e.g., individual schools) and **across groups** (e.g., school districts).
- Each group has its own parameters, but these parameters are drawn from a **higher-level distribution**.

**Example:**
Analyzing student test scores where students are nested within classrooms, and classrooms are nested within schools.

**Benefits of Hierarchical Modeling:**
- Accounts for variability at multiple levels.
- Improves estimates by borrowing strength across groups.
- Reduces bias in parameter estimates for grouped data.

---

<!-- TOC --><a name="q-what-is-design-of-experiments-doe"></a>
### Q: What is Design of Experiments (DOE)?

**A:** **Design of Experiments (DOE)** is a systematic approach to planning, conducting, and analyzing controlled experiments to understand the effects of multiple factors on a response variable. DOE helps identify relationships between inputs (factors) and outputs (responses) and optimize processes or products.

**Key Goals of DOE:**
- Determine which factors influence the outcome.
- Optimize performance by adjusting factor levels.
- Minimize variability and improve quality.

**Applications:**
- Manufacturing process optimization, product design, clinical trials, and marketing experiments.

---

<!-- TOC --><a name="q-what-are-the-core-principles-of-doe"></a>
### Q: What are the core principles of DOE?

**A:** The core principles of Design of Experiments (DOE) include:

1. **Replication:**
   - Repeating the experiment to estimate variability and improve accuracy.

2. **Randomization:**
   - Randomly assigning treatments to reduce bias and ensure the results are generalizable.

3. **Blocking:**
   - Grouping similar experimental units to control for known sources of variability.

4. **Factorial Design:**
   - Studying multiple factors simultaneously by testing all possible combinations of factor levels.

5. **Control:**
   - Using a control group or baseline condition for comparison with experimental conditions.

These principles help ensure experiments are rigorous, unbiased, and statistically valid.

---

<!-- TOC --><a name="q-what-is-ab-testing"></a>
### Q: What is A/B Testing?

**A:** **A/B testing** is a type of controlled experiment used to compare two versions (A and B) of a product, webpage, or feature to determine which one performs better. It is widely used in digital marketing, product design, and software development.

**Steps in A/B Testing:**

1. **Define the Goal:** Identify the key metric (e.g., conversion rate).
2. **Split the Audience:** Randomly assign users to **Group A** (control) and **Group B** (variant).
3. **Implement the Test:** Expose Group A to the original version and Group B to the modified version.
4. **Measure Outcomes:** Collect data for the key metric.
5. **Analyze Results:** Use statistical methods (e.g., hypothesis testing) to determine if the difference is significant.

**Example:**
Testing two versions of a website landing page to see which one leads to more sign-ups.

---

<!-- TOC --><a name="q-what-are-multi-armed-bandits"></a>
### Q: What are multi-armed bandits?

**A:** **Multi-armed bandits** are a type of sequential decision-making problem where an agent must choose between multiple options (called "arms") to maximize cumulative rewards over time. Each arm provides a random reward drawn from an unknown probability distribution. The term comes from the analogy of a gambler choosing between slot machines (bandits) in a casino.

**Goal:** Balance between exploring different arms to learn their rewards and exploiting the arm that appears to give the highest reward.

**Applications:**
- Online ad placement (choosing which ad to display).
- Clinical trials (choosing treatments).
- Dynamic pricing (optimizing price offers).

---

<!-- TOC --><a name="q-explain-the-exploration-exploitation-tradeoff-for-adaptive-environments"></a>
### Q: Explain the exploration-exploitation tradeoff for adaptive environments.

**A:** The **exploration-exploitation tradeoff** describes the balance between two strategies in decision-making for adaptive environments:

1. **Exploration:**
   - Trying new options to gain more information about their potential rewards.
   - Example: Testing new ads to discover their click-through rates.

2. **Exploitation:**
   - Choosing the best-known option to maximize immediate reward.
   - Example: Displaying the ad with the highest click-through rate so far.

**The Challenge:**
- **Too much exploration:** You may miss out on maximizing rewards by not taking advantage of known good options.
- **Too much exploitation:** You may fail to discover potentially better options.

**Example:**
In a **multi-armed bandit problem**, a gambler must decide how often to try different slot machines (exploration) versus playing the one that has given the highest payout so far (exploitation).

**Common Strategies to Balance Exploration and Exploitation:**
- **Epsilon-Greedy:** Explore randomly with probability \(\epsilon\), exploit otherwise.
- **Upper Confidence Bound (UCB):** Select the option with the highest upper confidence bound on expected reward.
- **Thompson Sampling:** Use Bayesian inference to sample actions based on their probability of being optimal.

---

<!-- TOC --><a name="q-what-are-ab-testing-metrics"></a>
### Q: What are A/B testing metrics?

**A:** **A/B testing metrics** are quantitative measures used to evaluate the effectiveness of variations in an A/B test. Common metrics include:

1. **Conversion Rate:** The percentage of users who complete a desired action (e.g., purchase, sign-up).
2. **Click-Through Rate (CTR):** The percentage of users who click on a link or button.
3. **Bounce Rate:** The percentage of users who leave a page without further interaction.
4. **Average Revenue Per User (ARPU):** The average income generated per user.
5. **Engagement Metrics:** Time spent on site, pages viewed, etc.
6. **Retention Rate:** The percentage of users who return after their first visit.
7. **Lift:** The relative improvement between the control and the variant.

---

<!-- TOC --><a name="q-what-is-statistical-power"></a>
### Q: What is statistical power?

**A:** **Statistical power** is the probability that a hypothesis test correctly rejects the null hypothesis ($H_0$) when the alternative hypothesis ($H_1$) is true. In other words, it measures the likelihood of detecting an actual effect.

$$
\text{Power} = 1 - \beta
$$

Where $\beta$ is the probability of a Type II error (false negative).

**Factors Affecting Power:**
- **Sample Size:** Larger samples increase power.
- **Effect Size:** Larger effects are easier to detect.
- **Significance Level ($\alpha$)**: Higher $\alpha$ increases power.
- **Variability:** Lower variability increases power.

A power of **0.8 (80%)** is commonly used, meaning there is an 80% chance of detecting a true effect.

---

<!-- TOC --><a name="q-what-are-tree-based-machine-learning-methods"></a>
### Q: What are tree-based machine learning methods?

**A:** **Tree-based machine learning methods** use decision trees as their core structure to make predictions. Common methods include:

1. **Decision Trees:** Simple models that split data based on feature values.
2. **Random Forest:** An ensemble of decision trees trained on random subsets of data and features.
3. **Gradient Boosting:** Sequentially builds trees to correct errors made by previous trees (e.g., XGBoost, LightGBM).
4. **Bagging (Bootstrap Aggregation):** Combines multiple decision trees to reduce variance.
5. **Extra Trees (Extremely Randomized Trees):** Similar to random forests but with randomized splits.
6. **Classification and Regression Trees (CART):** Used for both classification and regression tasks.

**Advantages:**
- Handle non-linear relationships.
- Perform well on structured/tabular data.
- Robust to outliers and missing data.

---

<!-- TOC --><a name="q-what-are-gradient-boosted-decision-trees-gbdt"></a>
### Q: What are gradient-boosted decision trees (GBDT)?

**A:** **Gradient-Boosted Decision Trees (GBDT)** are an ensemble learning method that builds a series of decision trees sequentially, where each new tree corrects errors made by the previous trees. It minimizes a loss function using **gradient descent** to improve performance iteratively.

**Key Features:**
- Trees are added one at a time.
- Each new tree focuses on reducing the residuals (errors) of previous trees.
- Effective for both classification and regression tasks.

**Advantages:**
- High predictive accuracy.
- Handles non-linear relationships well.

**Common Libraries:**
- **XGBoost**, **LightGBM**, **CatBoost**.

---

<!-- TOC --><a name="q-what-are-random-forests"></a>
### Q: What are random forests?

**A:** **Random Forests** are an ensemble learning method that builds multiple decision trees using different random subsets of the data and features. The final prediction is made by aggregating the outputs of all the trees (majority vote for classification, average for regression).

**Key Characteristics:**
- **Bagging (Bootstrap Aggregation):** Each tree is trained on a bootstrapped sample of the data.
- **Feature Randomness:** Each split considers a random subset of features.
- Reduces **overfitting** compared to a single decision tree.

**Advantages:**
- Robust to overfitting.
- Handles large datasets and high-dimensional data well.
- Works for both classification and regression tasks.

---

<!-- TOC --><a name="q-what-is-xgboost"></a>
### Q: What is XGBoost?

**A:** **XGBoost** (eXtreme Gradient Boosting) is an optimized, scalable implementation of gradient-boosted decision trees (GBDT). It is designed for speed, performance, and efficiency, making it a popular choice for machine learning competitions and real-world applications.

**Key Features:**
- **Regularization:** Uses \(L1\) (Lasso) and \(L2\) (Ridge) penalties to reduce overfitting.
- **Tree Pruning:** Stops growing trees when further splits do not improve performance.
- **Parallel Processing:** Optimized for multi-threading to handle large datasets quickly.
- **Handling Missing Values:** Automatically manages missing data.

**Applications:**
- Classification, regression, ranking, and time series forecasting.
- Frequently used in Kaggle competitions and industry projects.

---

---

<!-- TOC --><a name="q-explain-l1l2-regularization"></a>
### Q: Explain L1/L2 regularization.

**A:** **L1 and L2 regularization** are techniques used to prevent overfitting in machine learning models by adding a penalty to the loss function based on the model's coefficients.

- **L1 Regularization (Lasso):** Adds the **absolute values** of the coefficients as a penalty term:
  $$\text{Loss} = \text{Original Loss} + \lambda \sum_{i=1}^{n} |w_i|$$
  - Encourages **sparsity** by shrinking some coefficients to exactly zero.

- **L2 Regularization (Ridge):** Adds the **squared values** of the coefficients as a penalty term:
  $$\text{Loss} = \text{Original Loss} + \lambda \sum_{i=1}^{n} w_i^2$$
  - Encourages smaller coefficients but does not set them to zero.

- **Elastic Net:** Combines both L1 and L2 regularization:
  $$\text{Loss} = \text{Original Loss} + \lambda_1 \sum_{i=1}^{n} |w_i| + \lambda_2 \sum_{i=1}^{n} w_i^2$$

---

<!-- TOC --><a name="q-what-is-ridge-regression"></a>
### Q: What is ridge regression?

**A:** **Ridge regression** is a type of linear regression that uses **L2 regularization** to prevent overfitting by adding a penalty proportional to the sum of the squared coefficients.

**Ridge Regression Formula:**

```math
\hat{\beta} = \arg\min_{\beta} \left( \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2 \right)
```

Where:
- $\lambda$ is the regularization parameter (controls the strength of the penalty).
- Larger $\lambda$ values shrink coefficients more.

**Key Features:**
- Reduces the magnitude of coefficients but does **not** set them to zero.
- Useful when predictors are highly correlated.

---

<!-- TOC --><a name="q-what-is-lasso-regression"></a>
### Q: What is lasso regression?

**A:** **Lasso regression** (Least Absolute Shrinkage and Selection Operator) is a type of linear regression that uses **L1 regularization** to prevent overfitting by adding a penalty proportional to the sum of the absolute values of the coefficients.

**Lasso Regression Formula:**

```math
\hat{\beta} = \arg\min_{\beta} \left( \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j| \right)
```

Where:
- $\lambda$ is the regularization parameter (controls the strength of the penalty).
- Larger $\lambda$ values shrink coefficients, potentially to **exactly zero**.

**Key Features:**
- Performs **feature selection** by setting some coefficients to zero.
- Useful when dealing with high-dimensional datasets with many irrelevant features.

---

<!-- TOC --><a name="q-what-is-a-neural-network-nn"></a>
### Q: What is a neural network (NN)?

**A:** A **neural network (NN)** is a machine learning model inspired by the structure of the human brain. It consists of layers of **nodes (neurons)** connected by **weights**. These layers typically include:

1. **Input Layer:** Receives the input features.
2. **Hidden Layers:** Apply transformations using activation functions.
3. **Output Layer:** Produces the final prediction.

Each neuron computes a weighted sum of its inputs and passes it through an **activation function** (e.g., ReLU, sigmoid) to introduce non-linearity.

**Applications:** Image recognition, natural language processing, and time series forecasting.

---

<!-- TOC --><a name="q-what-is-a-convolutional-neural-network-cnn"></a>
### Q: What is a convolutional neural network (CNN)?

**A:** A **Convolutional Neural Network (CNN)** is a type of neural network designed for processing data with a grid-like structure, such as images. CNNs use **convolutional layers** to automatically and adaptively learn spatial hierarchies of features.

**Key Components:**
- **Convolutional Layers:** Apply filters (kernels) to extract features like edges, shapes, and textures.
- **Pooling Layers:** Reduce spatial dimensions (e.g., max pooling).
- **Fully Connected Layers:** Final layers for classification or regression.

**Applications:** Image classification, object detection, and computer vision tasks.

---

<!-- TOC --><a name="q-what-is-a-recurrent-neural-network-rnn"></a>
### Q: What is a recurrent neural network (RNN)?

**A:** A **Recurrent Neural Network (RNN)** is a type of neural network designed for sequential data. RNNs have connections that form cycles, allowing them to maintain a **memory of previous inputs**.

**Key Feature:**
- **Hidden State:** Maintains information about previous time steps, enabling the model to capture temporal dependencies.

**Limitations:**
- Difficulty learning long-term dependencies due to vanishing/exploding gradients.

**Applications:** Time series forecasting, language modeling, speech recognition.

---

<!-- TOC --><a name="q-explain-the-similarities-between-cnns-and-rnns"></a>
### Q: Explain the similarities between CNNs and RNNs.

**A:** Similarities between CNNs and RNNs include:

1. **Neural Network Structure:** Both are composed of neurons, weights, and activation functions.
2. **Layer Types:** Both can have multiple layers (e.g., convolutional, recurrent, or fully connected).
3. **Backpropagation:** Both are trained using backpropagation and gradient descent.
4. **Feature Extraction:** Both learn hierarchical features from the data.

---

<!-- TOC --><a name="q-explain-the-differences-between-cnns-and-rnns"></a>
### Q: Explain the differences between CNNs and RNNs.

**A:** Differences between CNNs and RNNs include:

1. **Data Type:**
   - **CNNs:** Designed for grid-like data (e.g., images).
   - **RNNs:** Designed for sequential data (e.g., time series, text).

2. **Architecture:**
   - **CNNs:** Use convolutional and pooling layers to capture spatial patterns.
   - **RNNs:** Use recurrent connections and hidden states to capture temporal patterns.

3. **Memory:**
   - **CNNs:** Process each input independently.
   - **RNNs:** Maintain a hidden state to retain information from previous inputs.

4. **Common Applications:**
   - **CNNs:** Image classification, object detection.
   - **RNNs:** Language modeling, speech recognition.

---

<!-- TOC --><a name="q-what-are-embeddings-in-the-context-of-nns"></a>
### Q: What are embeddings in the context of NNs?

**A:** **Embeddings** are dense vector representations of categorical or discrete data in neural networks. They map high-dimensional inputs (e.g., words, items) to lower-dimensional continuous vectors, capturing semantic relationships.

**Examples:**
- **Word Embeddings:** Represent words in a continuous vector space (e.g., Word2Vec, GloVe).
- **Item Embeddings:** In recommendation systems, products are mapped to vectors reflecting their similarity.

**Benefits:**
- Reduces dimensionality.
- Captures relationships and similarities between entities.

---

<!-- TOC --><a name="q-what-is-sequence-modeling-for-demand-prediction"></a>
### Q: What is sequence modeling for demand prediction?

**A:** **Sequence modeling for demand prediction** involves using models designed to handle sequential data (e.g., historical demand patterns) to forecast future demand.

**Common Techniques:**
- **Recurrent Neural Networks (RNNs):** Handle sequential dependencies.
- **Long Short-Term Memory (LSTM) Networks:** Capture long-term dependencies.
- **Transformer Models:** Use attention mechanisms for long-range dependencies.
- **Temporal Convolutional Networks (TCNs):** Apply convolutions over time series data.

**Applications:**
- Predicting product demand, inventory management, and supply chain forecasting.

---

<!-- TOC --><a name="q-what-is-the-concept-of-dimensionality-reduction"></a>
### Q: What is the concept of dimensionality reduction?

**A:** **Dimensionality reduction** is the process of reducing the number of features (dimensions) in a dataset while preserving as much information as possible. It simplifies data, reduces computational cost, and mitigates overfitting.

**Types of Dimensionality Reduction:**
1. **Linear Methods:** Principal Component Analysis (PCA).
2. **Non-Linear Methods:** t-SNE, UMAP.

**Benefits:**
- Visualization of high-dimensional data.
- Faster model training.
- Reducing noise in data.

---

<!-- TOC --><a name="q-what-is-principal-component-analysis-pca"></a>
### Q: What is Principal Component Analysis (PCA)?

**A:** **Principal Component Analysis (PCA)** is a linear dimensionality reduction technique that transforms data into a set of new orthogonal axes (principal components) that capture the most variance.

**Steps:**
1. Standardize the data.
2. Compute the covariance matrix.
3. Find the eigenvectors and eigenvalues.
4. Project data onto the principal components.

**Applications:**
- Data compression, visualization, and noise reduction.

---

<!-- TOC --><a name="q-what-is-t-sne"></a>
### Q: What is t-SNE?

**A:** **t-Distributed Stochastic Neighbor Embedding (t-SNE)** is a non-linear dimensionality reduction technique used for visualizing high-dimensional data in 2D or 3D.

**How it Works:**
- Converts pairwise distances between points into probabilities.
- Minimizes the divergence between high-dimensional and low-dimensional distributions.

**Key Features:**
- Preserves local structure.
- Effective for visualizing clusters.

**Applications:**
- Visualizing word embeddings, image data, and high-dimensional datasets.

---

<!-- TOC --><a name="q-what-are-some-common-feature-engineering-methods"></a>
### Q: What are some common feature engineering methods?

**A:** Common feature engineering methods include:

1. **Scaling and Normalization:**
   - Standardization (z-score), Min-Max Scaling.

2. **Encoding Categorical Variables:**
   - One-Hot Encoding, Label Encoding, Target Encoding.

3. **Handling Missing Data:**
   - Imputation (mean, median, mode), creating "missing" indicator features.

4. **Feature Transformation:**
   - Log Transform, Polynomial Features, Box-Cox Transform.

5. **Feature Interaction:**
   - Combining features (e.g., product of two features).

6. **Dimensionality Reduction:**
   - PCA, t-SNE, UMAP.

7. **Text Features:**
   - TF-IDF, Word Embeddings.

8. **Time-Based Features:**
   - Extracting day, month, season, or lag features.

---

<!-- TOC --><a name="q-what-is-catboost"></a>
### Q: What is CatBoost?

**A:** **CatBoost** is a gradient boosting algorithm developed by Yandex, designed to handle categorical features efficiently. It is based on decision trees and optimized for performance and ease of use.

**Key Features:**
- **Handles Categorical Data:** Automatically processes categorical features without extensive preprocessing.
- **Ordered Boosting:** Reduces overfitting by processing data in an order that avoids target leakage.
- **Efficient:** Fast training and inference.
- **Supports GPU Acceleration.**

**Applications:**
- Classification, regression, ranking tasks.

---

<!-- TOC --><a name="q-what-is-linear-programming"></a>
### Q: What is linear programming?

**A:** **Linear programming (LP)** is an optimization technique for maximizing or minimizing a linear objective function subject to linear constraints (equalities and inequalities).

**General Form:**

Maximize or Minimize:
```math
c_1x_1 + c_2x_2 + \ldots + c_nx_n
```

Subject to:
```math
a_{11}x_1 + a_{12}x_2 + \ldots + a_{1n}x_n \leq b_1
```
```math
x_i \geq 0 \quad \text{for all } i
```

**Applications:**
- Resource allocation, logistics, scheduling, manufacturing.

---

<!-- TOC --><a name="q-what-are-simplex-algorithms"></a>
### Q: What are simplex algorithms?

**A:** The **simplex algorithm** is a method for solving linear programming problems. It systematically explores the vertices of the feasible region to find the optimal solution.

**Key Steps:**
1. Start at an initial vertex.
2. Move to adjacent vertices that improve the objective function.
3. Repeat until no further improvement is possible.

**Properties:**
- Guaranteed to find the optimal solution if it exists.
- Efficient for many real-world problems, though worst-case complexity is exponential.

---

<!-- TOC --><a name="q-what-is-convex-optimization"></a>
### Q: What is convex optimization?

**A:** **Convex optimization** is a class of optimization problems where the objective function is convex, and the feasible region (defined by constraints) is a convex set. In convex problems, any local minimum is also a global minimum.

**General Form:**

Minimize:

```math
f(x)
```

Subject to:

```math
g_i(x) \leq 0 \quad \text{and} \quad h_j(x) = 0
```

Where $f(x)$ and $g_i(x)$ are convex functions, and $h_j(x)$ are affine functions.

**Applications:**
- Machine learning (e.g., SVMs), finance (portfolio optimization), engineering design, control systems.

---

<!-- TOC --><a name="q-what-are-gradient-based-methods"></a>
### Q: What are gradient-based methods?

**A:** **Gradient-based methods** are optimization techniques that use gradients (partial derivatives) to minimize or maximize an objective function. These methods iteratively update model parameters by moving in the direction of the **negative gradient** to minimize a loss function.

**General Update Rule:**

```math
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)
```

Where:
- $\theta_t$: Model parameters at step $t$.
- $\eta$: Learning rate (step size).
- $\nabla_\theta L(\theta_t)$: Gradient of the loss function with respect to $\theta$.

**Examples:**
- **Gradient Descent**
- **Stochastic Gradient Descent (SGD)**
- **Momentum**
- **Adam Optimizer**

---

<!-- TOC --><a name="q-what-is-stochastic-gradient-descent-sgd"></a>
### Q: What is stochastic gradient descent (SGD)?

**A:** **Stochastic Gradient Descent (SGD)** is a gradient-based optimization method that updates model parameters using a **random subset (mini-batch) of the data** instead of the entire dataset. This makes the updates faster and more computationally efficient.

**Update Rule:**

```math
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t; x_i, y_i)
```

Where:
- $(x_i, y_i)$: A single data point or mini-batch.
- $\eta$: Learning rate.

**Advantages:**
- Faster updates, especially for large datasets.
- Introduces noise, which can help escape local minima.

**Disadvantages:**
- Noisy convergence; may require more iterations.
- Learning rate tuning can be challenging.

---

<!-- TOC --><a name="q-what-is-the-adam-optimizer"></a>
### Q: What is the Adam optimizer?

**A:** The **Adam (Adaptive Moment Estimation) optimizer** is an adaptive gradient-based optimization method that combines the benefits of **momentum** and **RMSProp**. It adapts the learning rate for each parameter based on estimates of the first and second moments of the gradients.

**Update Rule:**
1. Compute moving averages of the gradients:
   ```math
   m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
   ```
   ```math
   v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
   ```
2. Bias correction:
   ```math
   \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
   ```
3. Parameter update:
   ```math
   \theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
   ```

Where:
- $\beta_1$ and $\beta_2$: Decay rates for moving averages (default $\beta_1 = 0.9$, $\beta_2 = 0.999$).
- $\eta$: Learning rate.
- $\epsilon$: Small constant to prevent division by zero.

**Advantages:**
- Adaptive learning rates for each parameter.
- Combines momentum and RMSProp for efficient updates.
- Suitable for large datasets and sparse gradients.

---

<!-- TOC --><a name="q-what-is-the-agile-methodology"></a>
### Q: What is the Agile methodology?

**A:** **Agile methodology** is a project management approach focused on iterative development, collaboration, flexibility, and delivering value incrementally. It emphasizes responding to change over following a strict plan. Agile breaks projects into small, manageable units of work, promoting continuous feedback and improvement.

**Key Principles (from the Agile Manifesto):**
1. Individuals and interactions over processes and tools.
2. Working software over comprehensive documentation.
3. Customer collaboration over contract negotiation.
4. Responding to change over following a plan.

---

<!-- TOC --><a name="q-what-is-scrum"></a>
### Q: What is Scrum?

**A:** **Scrum** is an Agile framework for managing complex projects, particularly software development. It focuses on iterative development through fixed-length cycles called **sprints** (usually 1-4 weeks). Scrum promotes teamwork, accountability, and continuous progress.

**Key Roles:**
- **Product Owner:** Defines the vision and priorities.
- **Scrum Master:** Facilitates the process and removes obstacles.
- **Development Team:** Delivers the work.

**Key Events:**
- **Sprint Planning:** Plan the work for the sprint.
- **Daily Stand-Up:** Short daily meeting to discuss progress.
- **Sprint Review:** Demonstrate completed work.
- **Sprint Retrospective:** Reflect on improvements.

---

<!-- TOC --><a name="q-what-is-kanban"></a>
### Q: What is Kanban?

**A:** **Kanban** is an Agile methodology that focuses on visualizing work, limiting work-in-progress (WIP), and optimizing flow. It uses a **Kanban board** to represent tasks and their status (e.g., To Do, In Progress, Done).

**Key Principles:**
1. **Visualize the Workflow:** Use a board with columns for each stage of work.
2. **Limit Work-in-Progress (WIP):** Restrict the number of tasks in progress to prevent overloading.
3. **Manage Flow:** Optimize the flow of tasks through the process.
4. **Continuous Improvement:** Regularly review and improve the process.

**Best for:** Teams needing flexibility and continuous delivery.

---

<!-- TOC --><a name="q-what-is-sprint-planning"></a>
### Q: What is sprint planning?

**A:** **Sprint planning** is a Scrum event where the team plans the work to be done in the upcoming sprint. It sets the sprint's goal and defines the tasks needed to achieve it.

**Key Elements:**
1. **Sprint Goal:** A concise objective for the sprint.
2. **Backlog Selection:** The team selects items from the product backlog to work on.
3. **Task Breakdown:** The team breaks backlog items into actionable tasks.
4. **Capacity Planning:** Consider team availability and velocity.

**Outcome:** A clear plan of what will be delivered by the end of the sprint.

---

### Q: Using an analogy of trees in a forest, describe Bayesian statistics.

**A**: Imagine you're walking through a dense forest and trying to determine the species of trees you’re seeing. You start with a **prior belief** about the types of trees in the forest based on your past experience — for example, you think it's mostly pine trees.

As you explore, you start observing more trees and collecting data. Each new tree you see provides **evidence** that helps refine your understanding of the forest's composition. For instance, if you encounter a mix of pine and oak trees, your belief starts to shift.

Bayesian statistics works like this process of updating beliefs. Let’s break it down:

1. **Prior Belief (Prior Probability):**
   Before you enter the forest, you have an initial belief about what types of trees are there (e.g., “80% pine trees, 20% oak trees”).

2. **New Observations (Likelihood):**
   As you walk, you observe trees. Each observation (e.g., “I see an oak tree”) represents new data that informs you about the forest.

3. **Updating Belief (Posterior Probability):**
   Based on the observations, you update your belief. If you see many oak trees, your belief may change to “It’s 50% pine trees and 50% oak trees.” This updated belief is the **posterior probability**.

4. **Bayes' Theorem:**
   The mathematical rule that governs this updating process. It combines the prior belief and the new evidence to produce the posterior belief.

**The Analogy Summarized:**
- **The Forest:** The real-world process or dataset you’re studying.
- **Prior Belief:** Your initial assumption about the types of trees.
- **Evidence:** The trees you observe.
- **Posterior Belief:** Your refined understanding of the forest after gathering evidence.

Bayesian statistics is like continuously adjusting your understanding of the forest as you gather more and more information.