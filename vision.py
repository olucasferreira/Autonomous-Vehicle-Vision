import cv2
import math
import logging
import numpy as np
import paho.mqtt.client as mqtt
import config

altura = 1.37  # 1.5
contador = 0
config.lostLeft = bool
config.lostRight = bool
config.lastPosition = 0
config.noLost = bool

# Conexão com o servidor
SERVER = "localhost"  # IP local Servidor Docker - Broker MQTT
PORTA = 1883

# Onde as mensagens serão publicadas no servidor
FAIXA = "controladorFuzzy/faixa"
VISAO = "controladorFuzzy/visao"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Conectado!")
    else:
        print("Erro conexão %d: ", rc)


# Criando estrutura básica do cliente
# Client_id deve ser uma chave primária qualquer que identifique O cliente no servidor
client = mqtt.Client(client_id="Visao", protocol=mqtt.MQTTv5)  # Cria uma nova instância com nome indicado.
client.on_connect = on_connect  # Verifica se cliente conectado

# Autenticacao padrao
client.username_pw_set(username="esp32", password="esp32")
print("Conectando ao servidor ", SERVER)
servidor = client.connect(SERVER, PORTA)


def canny(img):
    """
    Aplica a tranformação Canny Edge detection na imagem.
        Parameters:
            img: imagem de input.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 7
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    canny = cv2.Canny(blur, 400, 300)
    # canny = cv2.Canny(blur, 720, 420)
    return canny


def region_of_interest(img):
    """
    Seleciona a região de interesse e retorna uma máscara.
        Parameters:
            img: imagem de input.
    """
    height = img.shape[0]
    width = img.shape[1]
    mask = np.zeros_like(img)

    polygon = np.array([[(0, height * 1 / altura),
                         (width, height * 1 / altura),
                         (width, height),
                         (0, height),
                         ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hough_lines(img):
    """
    Aplica a função HoughLinesP e detecta as linhas da imagem.
    Parameters:
            img: imagem de input.
    """
    houghLines = cv2.HoughLinesP(img, 1, np.pi / 180, 10, np.array([]), minLineLength=70, maxLineGap=100)
    # houghLines = cv2.HoughLinesP(img, 1, np.pi/180, 10, np.array([]), minLineLength=7.5, maxLineGap=7.5)
    return houghLines


def display_lines(img, lines):
    """
    Retorna a imagem com as linhas marcadas.
        Parameters:
            img: imagem de input.
            lines: saída do função houghLines.
    """
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return img


def display_lines_average(img, lines):
    """
    Retorna a imagem com a média das linhas marcadas.
        Parameters:
            img: imagem de input.
            lines: saída do função houghLines.
    """
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 18)
    return img


def average_slop_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        logging.info('When No line_segment segments detected, retorna lane_lines.')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1 / 3
    left_region_boundary = width * (
                1 - boundary)  # segmento da linha da faixa esquerda deve estar 2/3 à esquerda da tela
    right_region_boundary = width * boundary  # segmento da linha da faixa direita deve estar 2/3 à esquerda da tela

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        lane_lines.append(make_points(frame, left_fit_average))

    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)
        lane_lines.append(make_points(frame, right_fit_average))

    # Validação para retornar se uma linha foi perdida, se sim, retorna qual
    if len(right_fit) > 0 and len(left_fit) > 0:
        config.noLost = True
    else:
        if len(left_fit) == 0:
            config.lostLeft = True
            config.noLost == False
        else:
            config.lostLeft = False
            config.noLost = False

        if len(right_fit) == 0:
            config.lostRight = True
            config.noLost = False
        else:
            config.lostRight = False
            config.noLost = False

    return lane_lines


def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height
    y2 = int(y1 * 1 / altura)

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))

    return [[x1, y1, x2, y2]]


def compute_steering_angle(frame, lane_lines):
    if len(lane_lines) == 0:
        logging.info('compute_steering_angle: No lane lines detected, do nothing')
        return -90

    height, width, _ = frame.shape
    if len(lane_lines) == 1:
        logging.debug('Only detected one lane line, just follow it. %s' % lane_lines[0])
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
    else:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        # camera_mid_offset_percent = 0.02 - valor antigo
        camera_mid_offset_percent = 0.00  # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
        mid = int(width / 2 * (1 + camera_mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid

    # encontre o ângulo de direção, que é o ângulo entre a direção de navegação até o final da linha central
    y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)  # ângulo (em radianos) para centrar a linha vertical
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # ângulo (em graus) para centrar a linha vertical
    steering_angle = angle_to_mid_deg + 90  # este é o ângulo de direção necessário para a roda dianteira inclinar

    return steering_angle


def display_heading_line(frame, steering_angle, line_color=(0, 255, 0), line_width=10):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape
    #### Ângulos #################
    # 0-89 graus: lado direito
    # 90 graus: centro
    # 91-180 graus: lado esquerdo
    ##############################
    steering_angle_radian = (steering_angle / (180.0 * math.pi))
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)
    # cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
    return heading_image


def steer(frame, lane_lines):
    logging.debug('steering...')
    curr_steering_angle = 90
    if len(lane_lines) == 0:
        logging.error('steer: 0')
        client.publish(VISAO, b"0")
        return frame

    new_steering_angle = compute_steering_angle(frame, lane_lines)

    '''Se o modelo perder uma linha a condição retornará um valor próprio para
    ajstar o carro em direção a linha perdida'''
    if config.noLost == True:
        config.lastPosition = new_steering_angle
        print("Última posição:", config.lastPosition)
    if config.lostLeft == True and config.lastPosition < 90:
        MENSAGEM = str(60)  # centroDireita da pista
        client.publish(VISAO, MENSAGEM)
        print("Perdeu Esquerda, está direita")
    elif config.lostLeft == True and config.lastPosition > 90:
        MENSAGEM = str(120)  # centroEsquerda da pista
        client.publish(VISAO, MENSAGEM)
        print("Perdeu Esquerda, está esquerda")
    elif config.lostRight == True and config.lastPosition > 90:
        MENSAGEM = str(120)  # controEsquerda da pista
        client.publish(VISAO, MENSAGEM)
        print("Perdeu Direita, está esquerda")
    elif config.lostRight == True and config.lastPosition < 90:
        MENSAGEM = str(60)  # controDireita da pista
        client.publish(VISAO, MENSAGEM)
        print("Perdeu Direita, está direita")
    else:
        # new_steering_angle = compute_steering_angle(frame, lane_lines)
        # print('Posição: 0-89 lado direito, 90 centro e 91-180 lado esquerdo. Ângulo: ', new_steering_angle)
        print('Posição: Ângulo: ', new_steering_angle)
        # Prepara a mensagem
        MENSAGEM = str(new_steering_angle)
        # Publica mensagem no tópico
        client.publish(VISAO, MENSAGEM)
    curr_heading_image = display_heading_line(frame, curr_steering_angle)
    return curr_heading_image

def faixa_pedestre(img):
    # coordenadas para cortar a imagem e identificar só a parte inferior
    y1 = 262
    y2 = 420
    x1 = 0
    x2 = 720
    img = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 7
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    ret, thresh = cv2.threshold(blur, 200, 280, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 5000:
            continue

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if len(contours) >= 6:
            cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
            MENSAGEM = str(999) # identifica a faixa de pedestre
            client.publish(FAIXA, MENSAGEM)

    return img


try:
    # Coloque o diretório do vídeo aqui
    capture = cv2.VideoCapture(0)
    # capture = cv2.VideoCapture("video_teste_3.mp4")
    # capture.set(cv2.CAP_PROP_FPS, 15)
    if not capture.isOpened():
        raise NameError('Vídeo não encontrado. Verifique o diretório.')

    else:
        while (capture.isOpened()):
            _, frame = capture.read()

            frame = cv2.resize(frame, (720, 420))
            canny_output = canny(frame)
            masked_output = region_of_interest(canny_output)
            faixa = faixa_pedestre(frame)
            lines = hough_lines(masked_output)
            average_lines = average_slop_intercept(frame, lines)
            line_image = display_lines_average(frame, average_lines)
            angle = compute_steering_angle(frame, average_lines)
            steering = steer(frame, average_lines)

            # Reproduz o vídeo em velocidade normal
            cv2.imshow('Detector de faixas', steering)
            # cv2.imshow(lines)
            if cv2.waitKey(60) & 0xFF == ord('r'):
                for i in range(20):
                    MENSAGEM = str(90)
                    client.publish(VISAO, MENSAGEM)

            if cv2.waitKey(60) & 0xFF == ord('q'):
                break

            # Para reproduzir o vídeo quadro a quadro, comente as linhas acima e descomente as duas linhas a seguir:
            # cv2.imshow('Detector de faixas', steering)
            # cv2.waitKey(0)
            contador = contador + 1
            # print("Contador:",contador)

except cv2.error as e:
    logging.warning('Ação cancelada.')

capture.release()
cv2.destroyAllWindows()
servidor = client.disconnect()
