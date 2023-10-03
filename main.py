from polygon import *


class Solid:
    def __init__(self, faces):
        self.polygons = []
        for f in faces:
            self.polygons.append(f)

    def render(self):
        for face in self.polygons:
            UpdateHandler.add_to_render_stack(face)

class RectPrismOrtho(Solid):
    def __init__(self, x, y, z, dx=1, dy=1, dz=1):
        self.faces = [Polygon([x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z]),
                 Polygon([x, y, z], [x+dx, y, z], [x+dx, y, z+dz], [x, y, z+dz]),
                 Polygon([x, y, z], [x, y+dy, z], [x, y + dy, z+dz], [x, y, z+dz]),
                 Polygon([x+dx, y+dy, z+dz], [x, y+dy, z+dz], [x, y, z+dz], [x+dx, y, z+dz]),
                 Polygon([x + dx, y + dy, z + dz], [x, y + dy, z + dz], [x, y+dy, z], [x + dx, y+dy, z]),
                 Polygon([x + dx, y + dy, z + dz], [x+dx, y, z + dz], [x+dx, y, z], [x + dx, y + dy, z])]
        super().__init__(self.faces)


if __name__ == '__main__':
    # poly = Polygon([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]])
    # other_poly = Polygon([[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]])
    # poly3 = Polygon([0, 0, 3], [0, 1, 3], [1, 1, 3], [1, 0, 3])
    # poly4 = Polygon([0, 0, 4], [0, -5, 4], [-5, -5, 4], [-5, 0, 4])
    # UpdateHandler.add_to_render_stack(poly3)
    # UpdateHandler.add_to_render_stack(poly4)
    # UpdateHandler.add_to_render_stack(poly)
    # UpdateHandler.add_to_render_stack(other_poly)
#    cube = RectPrismOrtho(1, -0.5, 1, 1, 1, 1)
    cube = RectPrismOrtho(0, 0, 0, 1, 1, 1)
    cube.render()
    UpdateHandler.render()
    tk.mainloop()
