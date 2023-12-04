import numpy as np
import PIL.Image
import PIL.ImageOps
import pygame
from pygame.surfarray import array3d
from skorch import NeuralNetClassifier

from model import SimpleCNN

WIDTH, HEIGHT = 300, 400

pygame.init()
fps_clock = pygame.time.Clock()

screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption("MNIST handwritten classification")
font = pygame.font.Font(pygame.font.get_default_font(), 20)
running = True
drawing = False
screen.fill((255, 255, 255))
prediction = 0
area = (10, 100, 280, 280)
model = NeuralNetClassifier(SimpleCNN)

model.initialize()
model.load_params(f_params="checkpoints/model-adamax.pkl")

while running:
    pygame.draw.rect(screen, pygame.color.Color(0x0C7746), area, 1)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONUP:
            y = PIL.ImageOps.invert(
                PIL.Image.fromarray(
                    array3d(screen.subsurface(area)).transpose((1, 0, 2))
                ).resize((28, 28))
            ).convert("L")
            y = np.array(y).reshape((1, 1, 28, 28)).astype(np.float32)
            y /= 255.0
            prediction = model.predict(y)[0]
            text_surface = font.render(f"Prediction: {prediction}", True, (0, 0, 0))
            screen.fill((255, 255, 255), (0, 0, 300, 100))
            screen.blit(text_surface, dest=(0, 0))
            drawing = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DELETE:
                screen.fill((255, 255, 255))

    if drawing:
        pos = pygame.mouse.get_pos()
        if (
            area[0] < pos[0] < area[0] + area[2]
            and area[1] < pos[1] < area[1] + area[3]
        ):
            pygame.draw.circle(screen, (0, 0, 0), pos, 7)

    pygame.display.flip()
    fps_clock.tick(120)

pygame.quit()
