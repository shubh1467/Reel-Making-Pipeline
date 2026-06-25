import cv2

cap = cv2.VideoCapture("/Users/takneekmacmini/Documents/Reels Pipeline/Reel-Making-Pipeline/raw_slowed_0.5x.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

print("fps =", fps)
print("frames =", frames)
print("duration =", frames/fps)