# xgbLosses
Class of custom loss functions that can be used for XGBoost training.

Example of function that can be passed as value for the 'objective" hyperparameter:

```
from xgbLosses_sklearn import xgbLosses_sklearn

xgbLosses_sklearn(alpha=2.0, gamma=0.3).composite
```
