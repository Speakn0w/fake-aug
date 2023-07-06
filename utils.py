import sys

def print_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    sys.stdout.flush()


def logger(info):
    fold, epoch = info['fold'], info['epoch']
    if epoch == 1 or epoch % 10 == 0:
        train_acc, test_acc, train_loss, train_label_loss = info['train_acc'], info['test_acc'], info['train_loss'], info['train_label_loss']
        res = info['test_f1']
        val_acc, val_f1_U, val_f1_NR, val_f1_T, val_f1_F = res["accuracy"], res["f1_U"], res["f1_NR"], res["f1_T"], res[
            "f1_F"]
        #train_acc, test_acc= info['train_acc'], info['test_acc']
        print('{:02d}/{:03d}: Train Acc: {:.3f}, Test Accuracy: {:.3f}, Test f1_U: {:.4f}, Test f1_NR: {:.4f}, Test f1_T: {:.4f}, Test f1_F: {:.4f},Train Loss: {:.4f}, Train_label Loss: {:.4f}'.format(
            fold, epoch, train_acc, test_acc, val_f1_U, val_f1_NR, val_f1_T, val_f1_F, train_loss, train_label_loss))

        # print('{:02d}/{:03d}: Train Acc: {:.3f}, Test Accuracy: {:.3f}'.format(
        #      fold, epoch, train_acc, test_acc))

    sys.stdout.flush()
def logger_binary(info):
    fold, epoch = info['fold'], info['epoch']
    if epoch == 1 or epoch % 10 == 0:
        train_acc, test_acc, train_loss, train_label_loss = info['train_acc'], info['test_acc'], info['train_loss'], info['train_label_loss']
        res = info['test_f1']
        val_acc, val_f1_U, val_f1_NR, val_f1_T, val_f1_F = res["accuracy"], res["f1_U"], res["f1_NR"], res["f1_T"], res[
            "f1_F"]
        #train_acc, test_acc= info['train_acc'], info['test_acc']
        print('{:02d}/{:03d}: Train Acc: {:.3f}, Test Accuracy: {:.3f}, Test f1_U: {:.4f}, Test f1_NR: {:.4f}, Test prec: {:.4f}, Test rec: {:.4f},Train Loss: {:.4f}, Train_label Loss: {:.4f}'.format(
            fold, epoch, train_acc, test_acc, val_f1_U, val_f1_NR, val_f1_T, val_f1_F, train_loss, train_label_loss))

        # print('{:02d}/{:03d}: Train Acc: {:.3f}, Test Accuracy: {:.3f}'.format(
        #      fold, epoch, train_acc, test_acc))

    sys.stdout.flush()

def logger_gcn(info):
    fold, epoch = info['fold'], info['epoch']
    if epoch == 1 or epoch % 10 == 0:
        train_acc, test_acc, train_loss = info['train_acc'], info['test_acc'], info['train_loss']
        res = info['test_f1']
        val_acc, val_f1_U, val_f1_NR, val_f1_T, val_f1_F = res["accuracy"], res["f1_U"], res["f1_NR"], res["f1_T"], res[
            "f1_F"]
        #train_acc, test_acc= info['train_acc'], info['test_acc']
        print('{:02d}/{:03d}: Train Acc: {:.3f}, Test Accuracy: {:.3f}, Test f1_U: {:.4f}, Test f1_NR: {:.4f}, Test f1_T: {:.4f}, Test f1_F: {:.4f},Train Loss: {:.4f}'.format(
                fold, epoch, train_acc, test_acc, val_f1_U, val_f1_NR, val_f1_T, val_f1_F, train_loss))

        # print('{:02d}/{:03d}: Train Acc: {:.3f}, Test Accuracy: {:.3f}'.format(
        #      fold, epoch, train_acc, test_acc))

    sys.stdout.flush()

def logger_gcn_binary(info):
    fold, epoch = info['fold'], info['epoch']
    if epoch == 1 or epoch % 10 == 0:
        train_acc, test_acc, train_loss = info['train_acc'], info['test_acc'], info['train_loss']
        res = info['test_f1']
        val_acc, val_f1_U, val_f1_NR, val_f1_T, val_f1_F = res["accuracy"], res["f1_U"], res["f1_NR"], res["f1_T"], res[
            "f1_F"]
        #train_acc, test_acc= info['train_acc'], info['test_acc']
        print('{:02d}/{:03d}: Train Acc: {:.3f}, Test Accuracy: {:.3f}, Test f1_U: {:.4f}, Test f1_NR: {:.4f}, Test prec: {:.4f}, Test rec: {:.4f},Train Loss: {:.4f}'.format(
                fold, epoch, train_acc, test_acc, val_f1_U, val_f1_NR, val_f1_T, val_f1_F, train_loss))

        # print('{:02d}/{:03d}: Train Acc: {:.3f}, Test Accuracy: {:.3f}'.format(
        #      fold, epoch, train_acc, test_acc))

    sys.stdout.flush()


