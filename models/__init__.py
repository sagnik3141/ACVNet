from models.acv import ACVNet
from models.acv_spc_multi import ACVSPC
from models.loss import model_loss_train_attn_only, model_loss_train_freeze_attn, model_loss_train, model_loss_test, model_loss_train_spc

__models__ = {
    "acvnet": ACVNet,
    "acvspc": ACVSPC
}
