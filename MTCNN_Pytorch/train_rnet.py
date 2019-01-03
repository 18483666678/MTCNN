import nets
import train01
import os


if __name__ == '__main__':
    net = nets.RNet()
    if not os.path.exists("param02"):
        os.makedirs("param02")

    trainer = train01.Trainer(net,"param02/rnet.pt",r"E:\celeba1208\24")
    trainer.train()