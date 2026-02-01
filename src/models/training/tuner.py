import optuna 
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import logging


logger = logging.getLogger(__name__)

class ChurnTuner:
    def __init__(self,X, y):
        self.X = X
        self.y = y
    
    def objective(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate',0.01, 0.2,log=True),
            'subsample': trial.suggest_float('subsample',0.5,1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight',1.0,2.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree',0.5,1.0),
            'eval_metric': 'logloss',
            'random_state': 42
        }

        model = XGBClassifier(**params)

        score = cross_val_score(model, self.X, self.y, cv=3, scoring='f1').mean()
        logger.info(f"Score for trial: {score}")
        return score

    def optimize(self, n_trials=50):
        logger.info(f"Starting hyperparameter optimization with {n_trials} trails...")
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)

        logger.info(f"Best trial: {study.best_trial.params} with value: {study.best_trial.value}")
        return study.best_trial.params