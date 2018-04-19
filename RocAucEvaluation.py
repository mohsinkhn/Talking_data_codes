from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, batch_size=10000, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            logs["val_score"] = score
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))