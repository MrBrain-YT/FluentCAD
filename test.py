import numpy as np

# 1 elipse
center1 = np.array([-1.03923048,  0.51961524])
big_radius1 = 0.17320508075688779
small_radius1 = 0.17320508073956728
angle1 = 0

# 2 elipse
center2 = np.array([-1.03923048,  0.34641016])
big_radius2 = 0.17320508075688779
small_radius2 = 0.08487433002981015
angle2 = -90


# elipse_thetta = np.atan2(0, -1)
# elipse_angle = (elipse_thetta * 180 / np.pi)
# print(elipse_angle)

old_f = None
points = np.linspace(0, 2*np.pi, 200)
is_plus = True
for index, t in enumerate(points):
    x1 = center1[0] + big_radius1 * np.cos(t) * np.cos(np.radians(angle1)) - small_radius1 * np.sin(t) * np.sin(np.radians(angle1))
    y1 = center1[1] + big_radius1 * np.cos(t) * np.sin(np.radians(angle1)) + small_radius1 * np.sin(t) * np.cos(np.radians(angle1))
    x2 = center2[0] + big_radius2 * np.cos(t) * np.cos(np.radians(angle2)) - small_radius2 * np.sin(t) * np.sin(np.radians(angle2))
    y2 = center2[1] + big_radius2 * np.cos(t) * np.sin(np.radians(angle2)) + small_radius2 * np.sin(t) * np.cos(np.radians(angle2))
    
    # f(t) = ((x(t)-cx2)·cos θ2 + (y(t)-cy2)·sin θ2)² / a2² + (-(x(t)-cx2)·sin θ2 + (y(t)-cy2)·cos θ2)² / b2²  − 1,
    f = ((x1 - center2[0]) * np.cos(np.radians(angle2)) + (y1 - center2[1]) * np.sin(np.radians(angle2)))**2 / big_radius2**2 + \
        (-(x1 - center2[0]) * np.sin(np.radians(angle2)) + (y1 - center2[1]) * np.cos(np.radians(angle2)))**2 / small_radius2**2 - 1
        
    if old_f is not None:
        if (f < 0) != (old_f < 0):
            old_f = f
            print("Знак изменён")
            minimum_point = None
            old_f2 = None
            for t2 in np.linspace(points[index-1], t, 200):
                x1 = center1[0] + big_radius1 * np.cos(t2) * np.cos(np.radians(angle1)) - small_radius1 * np.sin(t2) * np.sin(np.radians(angle1))
                y1 = center1[1] + big_radius1 * np.cos(t2) * np.sin(np.radians(angle1)) + small_radius1 * np.sin(t2) * np.cos(np.radians(angle1))
                x2 = center2[0] + big_radius2 * np.cos(t2) * np.cos(np.radians(angle2)) - small_radius2 * np.sin(t2) * np.sin(np.radians(angle2))
                y2 = center2[1] + big_radius2 * np.cos(t2) * np.sin(np.radians(angle2)) + small_radius2 * np.sin(t2) * np.cos(np.radians(angle2))
                
                # f(t) = ((x(t)-cx2)·cos θ2 + (y(t)-cy2)·sin θ2)² / a2² + (-(x(t)-cx2)·sin θ2 + (y(t)-cy2)·cos θ2)² / b2²  − 1,
                f2 = ((x1 - center2[0]) * np.cos(np.radians(angle2)) + (y1 - center2[1]) * np.sin(np.radians(angle2)))**2 / big_radius2**2 + \
                    (-(x1 - center2[0]) * np.sin(np.radians(angle2)) + (y1 - center2[1]) * np.cos(np.radians(angle2)))**2 / small_radius2**2 - 1
                # print(f)
                if old_f2 is not None:
                    if (f2 < 0) != (old_f2 < 0):
                        old_f2 = f2
                        minimum_point = (float(x1), float(y1))
                if old_f2 is None:
                    old_f2 = f2
            print(f"Intersection point: {minimum_point}, f={old_f2}")
    if old_f is None:
        old_f = f
    
    # print(f)
    # if f == 0:
    #     print(f"Intersection point: ({x1}, {y1})")
        
    old_f = f



# import numpy as np

# # 1 ellipse
# center1 = np.array([-1.38564065, 0])
# big_radius1 = 0.34641016151377557
# small_radius1 = 0.16514009915302438
# angle1 = 0

# # 2 ellipse
# center2 = np.array([-0.8660254, 0])
# big_radius2 = 0.34641016151377557
# small_radius2 = 0.17320508073956725
# angle2 = 0

# def ellipse_point(t, center, a, b, angle_deg):
#     angle = np.radians(angle_deg)
#     x = center[0] + a * np.cos(t) * np.cos(angle) - b * np.sin(t) * np.sin(angle)
#     y = center[1] + a * np.cos(t) * np.sin(angle) + b * np.sin(t) * np.cos(angle)
#     return x, y

# def ellipse_f(x, y, center, a, b, angle_deg):
#     angle = np.radians(angle_deg)
#     dx, dy = x - center[0], y - center[1]
#     lx =  dx * np.cos(angle) + dy * np.sin(angle)
#     ly = -dx * np.sin(angle) + dy * np.cos(angle)
#     return (lx / a) ** 2 + (ly / b) ** 2 - 1

# def bisect(t0, t1, center1, a1, b1, angle1, center2, a2, b2, angle2, iterations=50):
#     x0, y0 = ellipse_point(t0, center1, a1, b1, angle1)
#     f0 = ellipse_f(x0, y0, center2, a2, b2, angle2)
    
#     lo, hi = t0, t1
#     for _ in range(iterations):
#         mid = (lo + hi) / 2
#         xm, ym = ellipse_point(mid, center1, a1, b1, angle1)
#         fm = ellipse_f(xm, ym, center2, a2, b2, angle2)
#         if f0 * fm < 0:
#             hi = mid
#         else:
#             lo = mid
#             f0 = fm
    
#     tm = (lo + hi) / 2
#     return ellipse_point(tm, center1, a1, b1, angle1)

# # --- Основной поиск ---
# STEPS = 200
# EPS = 1e-6  # порог дубликатов
# found_points = []

# ts = np.linspace(0, 2 * np.pi, STEPS, endpoint=False)  # endpoint=False — не повторяем t=0

# f_values = []
# for t in ts:
#     x, y = ellipse_point(t, center1, big_radius1, small_radius1, angle1)
#     f_values.append(ellipse_f(x, y, center2, big_radius2, small_radius2, angle2))

# for i in range(len(ts)):
#     f0 = f_values[i]
#     f1 = f_values[(i + 1) % len(ts)]
    
#     if f0 * f1 < 0:  # смена знака
#         t0 = ts[i]
#         t1 = ts[(i + 1) % len(ts)]
        
#         # Последний шаг замыкает круг: t[last] → t[0]+2π
#         if i == len(ts) - 1:
#             t1 = ts[0] + 2 * np.pi
        
#         point = bisect(t0, t1, center1, big_radius1, small_radius1, angle1,
#                                 center2, big_radius2, small_radius2, angle2)
        
#         # Проверка на дубликат
#         is_duplicate = any(
#             np.hypot(point[0] - p[0], point[1] - p[1]) < EPS
#             for p in found_points
#         )
        
#         if not is_duplicate:
#             found_points.append(point)
#             print(f"Точка пересечения: ({point[0]:.8f}, {point[1]:.8f})")

# print(f"\nВсего точек: {len(found_points)}")