# PyGame的学习



## 1 安装

```
python3 -m pip install -U pygame --user
```



## 2 Tutorials

### 2.1 Introduction to Pygame

接下来我们首先看一个最简单的弹力球的模型

```python
import sys, pygame

pygame.init()

size = width, height = 600, 480
speed = [2, 2]
black = 0, 0, 0

screen = pygame.display.set_mode(size)

ball = pygame.image.load("intro_ball.gif")
ballrect = ball.get_rect()

while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

    ballrect = ballrect.move(speed)
    if ballrect.left < 0 or ballrect.right > width:
        speed[0] = -speed[0]
    if ballrect.top < 0 or ballrect.bottom > height:
        speed[1] = -speed[1]

    screen.fill(black)
    screen.blit(ball, ballrect)
    pygame.display.flip()
```

<img src="C:\Users\edj\AppData\Roaming\Typora\typora-user-images\image-20200923213342623.png" alt="image-20200923213342623" style="zoom:50%;" />

















### 2.2 Import and Initialize



### 2.3 How do I move an Image?



### 2.4 Chimp Tutorials,Line by Line





### 2.5 Sprite Module Introduction





