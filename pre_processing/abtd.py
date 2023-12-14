N = 0 # numero de camadas UH
W = 0 # altura da imagem
H = 0 # largura da imagem

def calc_s(img, pixel_pos):
    #o calculo estava errado falta implementar esse aqui
    pass


def calc_ci_and_max(img, current_pixel_pos, a = 5, b = 5):
    s_pixel = calc_s(img, current_pixel_pos)
    s_pixel_neighboor_list = []
    for i in range(-a, a):
        for j in range(-b, b):
            s_pixel_neighboor = calc_s(img, [i+a, j+b])
    if s_pixel >= max(s_pixel_neighboor.append(s_pixel_neighboor)):
        return 1
    else:
        return 0

def calc_r(img, current_pixel_pos):
    r = calc_ci_and_max(img, current_pixel_pos)
    return r

def calc_fci(img, current_pixel_pos):
    w, h = img.shape
    for i in range(len(w)):
        for j in range(len(h)):
            calc_r(img, current_pixel_pos)


def calc_abtd(img):
    w, h = img.shape
    for y in range(w):
        for x in range(h):
            current_pixel_pos = [y,x]
            pi_num = calc_fci(img, current_pixel_pos)
            acumu = 0
            for i in range(len(N)):
                acumu += calc_fci(img, current_pixel_pos)
            pid_den = acumu
            pi = pi_num/pid_den