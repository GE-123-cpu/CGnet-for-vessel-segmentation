import argparse
import cv2
from torch import nn, optim
from tqdm import tqdm
from model import *
from dataset import *
from model import *
from loss import *
from utils import *

# from val import *


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--step', type=int, default=20,
                    help='val step')
parser.add_argument('--batch_size', type=float, default=1, help='batch size')
parser.add_argument('--EPOCHS', type=float, default=200, help='epochs')
parser.add_argument('--model_dir', type=str, default='./model_save/',
                    help='save path of model')
parser.add_argument('--raw_dir', type=str, default='./ROSE-2/train/8bit_original/',
                    help='raw image path in training')
parser.add_argument('--gt_dir', type=str, default='./ROSE-2/train/8bit_gt/',
                    help='gt image path in training')
parser.add_argument('--test_raw_dir', type=str, default='./ROSE-2/test/8bit_original/',
                    help='raw image path in testing')
parser.add_argument('--test_gt_dir', type=str, default='./ROSE-2/test/gt/',
                    help='gt image path in testing')
parser.add_argument('--save_prob_dir', type=str, default='./ROSE-2/predict/prob/',
                    help='save prob image path in testing')
parser.add_argument('--save_pred_dir', type=str, default='./ROSE-2/predict/pred/',
                    help='save pred image path in testing')

opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dice1_loss = dice_loss().to(device)
criterion1 = nn.L1Loss().to(device)
criterion = nn.MSELoss().to(device)


def visual_results(loss_arr, auc_arr,
                   writer, epoch, flag=""):
    loss_mean = loss_arr.mean()
    auc_mean = auc_arr.mean()

    writer.add_scalars("val" + flag + "_loss", {"val" + flag + "_loss": loss_mean}, epoch)
    writer.add_scalars("val" + flag + "_auc", {"val" + flag + "_auc": auc_mean}, epoch)

    return auc_mean


def adjust_learning_rate_poly(epoch, num_epochs, base_lr, power):
    lr = base_lr * (1 - epoch / num_epochs) ** power
    # for param_group in optimizer.param_groups:
    #    param_group['lr'] = lr
    return lr


def train_model(model, optimizer, dataloaders, num_epochs=opt.EPOCHS):
    # best_model = model
    best_fusion = {"epoch": 0, "auc": 0}
    # lr = adjust_learning_rate(epoch)  # - 1)
    # model.train()
    for epoch in range(num_epochs):
        model.train()
        lr = adjust_learning_rate_poly(epoch, num_epochs=opt.EPOCHS, base_lr=5e-4, power=0.9)
        # lr = adjust_learning_rate(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)
        # dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in tqdm(dataloaders):
            step += 1
            # x1 = torch.cat([x, x, x], dim=1)
            inputs = Variable(x.to(device))
            # inputs1 = Variable(z.to(device))
            labels = Variable(y.to(device))
            optimizer.zero_grad()
            a1, a2, a3, a4 = model(inputs)
            loss1 = criterion(a1, labels)
            loss2 = criterion(a2, labels)
            loss3 = criterion(a3, labels)
            loss4 = criterion(a4, labels)
            loss11 = dice1_loss(a1, labels)
            loss22 = dice1_loss(a2, labels)
            loss33 = dice1_loss(a3, labels)
            loss44 = dice1_loss(a4, labels)

            if epoch < 50:
                loss = loss22 + loss11 + loss33 + loss44
            # elif epoch > 50:
            #     loss = loss4
            else:
                loss = loss1 + loss2 + loss3 + loss4  #####ROSE2的loss方案

            # loss = loss1 + loss2 + loss3 + loss4
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss / step))
        if (epoch + 1) % opt.step == 0:
            model.eval()
            auc_lst = []
            test_dataset = DatasetFromFolder_test(opt.test_raw_dir, opt.test_gt_dir)
            dataloaders_test = DataLoader(test_dataset, batch_size=1)
            with torch.no_grad():
                for index, (a, c) in enumerate(dataloaders_test):
                    # x = torch.cat((x, x, x), dim=1)
                    a = Variable(a.to(device))
                    _, _, _, b = model(a)


                    img_y = b.squeeze().cpu().numpy()
                    pred_img = np.array(img_y * 255, np.uint8)
                    # thresh_value, thresh_pred_img = cv2.threshold(pred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    #pred_arr = img_y.squeeze().cpu().numpy()
                    gt_arr = c.squeeze().cpu().numpy()
                    auc_lst.append(calc_auc(pred_img, gt_arr))
                    auc_arr = np.array(auc_lst)

            fusion_auc = auc_arr.mean()
            mkdir(opt.model_dir)
            # if thin_auc >= best_thin["auc"]:
            #     best_thin["epoch"] = epoch + 1
            #     best_thin["auc"] = thin_auc
            #     torch.save(net.state_dict(), os.path.join(models_dir, "front_model-best_thin.pth"))
            # if thick_auc >= best_thick["auc"]:
            #     best_thick["epoch"] = epoch + 1
            #     best_thick["auc"] = thick_auc
            #     torch.save(net.state_dict(), os.path.join(models_dir, "front_model-best_thick.pth"))
            torch.save(model.state_dict(), opt.model_dir + str(epoch + 1) + "_model-best_fusion.pth")
            if fusion_auc >= best_fusion["auc"]:
                best_fusion["epoch"] = epoch + 1
                best_fusion["auc"] = fusion_auc
            torch.save(model.state_dict(), os.path.join(opt.model_dir, "front_model-best_fusion.pth"))
            # print("best thin: epoch %d\tauc %.4f" % (best_thin["epoch"], best_thin["auc"]))
            # print("best thick: epoch %d\tauc %.4f" % (best_thick["epoch"], best_thick["auc"]))
            print("best fusion: epoch %d\tauc %.4f" % (best_fusion["epoch"], best_fusion["auc"]))

    return model


def train_model1(model, optimizer, dataloaders, num_epochs=opt.EPOCHS):
    best_model = model
    min_loss = 1
    # lr = adjust_learning_rate(epoch)  # - 1)
    for epoch in range(num_epochs):
        model.train()
        lr = adjust_learning_rate_poly(epoch, num_epochs=opt.EPOCHS, base_lr=5e-4, power=0.9)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)
        # dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in tqdm(dataloaders):
            step += 1
            inputs = Variable(x.to(device))
            labels = Variable(y.to(device))
            # labels_thick = Variable(z.to(device))
            # labels_thin = Variable(w.to(device))

            # criterion = dicloss
            # input_layer2 = copy.deepcopy(inputs)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            a1, a2, a3, a4 = model(inputs)
            loss1 = criterion(a1, labels)
            loss2 = criterion(a2, labels)
            loss3 = criterion(a3, labels)
            loss4 = criterion(a4, labels)
            loss11 = dice1_loss(a1, labels)
            loss22 = dice1_loss(a2, labels)
            loss33 = dice1_loss(a3, labels)
            loss44 = dice1_loss(a4, labels)

            # if epoch < 20:
            #     loss = 0.5 * loss22 + loss11 + 0.5 * loss33
            # elif epoch > 40:
            #     loss = loss2 + loss3
            # else:
            #     loss = loss1 + 0.5 * loss2 + 0.5 * loss3  #####ROSE2的loss方案
            # loss = loss3 + 0.5 * loss2 + loss1

            # if epoch < 20:
            #     loss = loss22 + loss11 + loss33 + loss44
            # # # elif epoch > 40:
            # # #     loss = loss4
            # else:
            #     loss = loss1 + loss2 + loss3 + loss4  #####ROSE2的loss方案

            loss = loss1 + loss2 + loss3 + loss4
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss / step))
        if (epoch_loss / step) < min_loss:
            min_loss = (epoch_loss / step)
            best_model = model
    torch.save(best_model.state_dict(), opt.model_dir)
    return best_model


def train():
    model = segnet().to(device)
    # model1 = fusion().to(device)
    batch_size = 1
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = bce_loss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=0.0001)
    train_dataset = DatasetFromFolder2(LR_image_dir=opt.raw_dir,
                                       HR_image_dir=opt.gt_dir
                                       )
    dataloaders = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    train_model(model, optimizer, dataloaders)


def test():
    model = segnet()
    model.load_state_dict(torch.load(os.path.join(opt.model_dir, "front_model-best_fusion.pth")))
    #model.load_state_dict(torch.load(os.path.join(opt.model_dir, "front_model-80-0.8562.pth")))
    test_dataset = DatasetFromFolder1(opt.test_raw_dir)
    dataloaders = DataLoader(test_dataset, batch_size=1)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        for index, x in enumerate(dataloaders):
            _, _, _, y = model(x)
            img_y = torch.squeeze(y).numpy()
            pred_img = np.array(img_y * 255, np.uint8)
            thresh_value, thresh_pred_img = cv2.threshold(pred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            io.imsave(opt.save_prob_dir + str(index) + "_predict.png", img_y)
            io.imsave(opt.save_pred_dir + str(index) + "_predict.png", thresh_pred_img)
            plt.pause(0.01)
        plt.show()


if __name__ == '__main__':
    print("开始训练")
    train()
    print("训练完成，保存模型")
    print("开始预测")
    test()
