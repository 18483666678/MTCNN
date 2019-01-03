import nets
import train01
import os

if __name__ == '__main__':
    net = nets.PNet()

    if not os.path.exists("param02"):
        os.makedirs("param02")

    trainer = train01.Trainer(net, 'param02/pnet.pt', r"E:\celeba1208\12")
    trainer.train()
