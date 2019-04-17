if __name__ == '__main__':
    with open('../datasets/yoho1_64/train.txt', 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write('writer:Fatsheep\n' + content)
