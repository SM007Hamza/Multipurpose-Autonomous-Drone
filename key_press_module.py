import pygame


def init():
    pygame.init()
    win = pygame.display.set_mode((400, 400))


def get_Key(KeyName):
    ans = False
    for eve in pygame.event.get():
        pass
    KeyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, 'K_{}'.format(KeyName))
    if KeyInput[myKey]:
        ans = True
    pygame.display.update()

    return ans


def main():
    if get_Key("LEFT"):
        print("Left Key Pressed")
    if get_Key("RIGHT"):
        print("Right Key Pressed")


if __name__ == '__main__':
    init()
    while True:
        main()
