"""Mapowanie etykiet gestów HaGRIDv2 na liczbę wyprostowanych palców (wersja robocza, sprawdzana w 02_eda)."""

GESTURE_TO_COUNT = {
    "fist": 0,
    "one": 1, "point": 1, "like": 1, "dislike": 1, "mute": 1, "little_finger": 1, "middle_finger": 1,
    "peace": 2, "peace_inverted": 2, "two_up": 2, "two_up_inverted": 2, "rock": 2, "call": 2,
    "thumb_index": 2, "thumb_index2": 2,
    "three": 3, "three2": 3, "three3": 3, "ok": 3, "three_gun": 3,
    "four": 4,
    "palm": 5, "stop": 5, "stop_inverted": 5,
}

# Gesty dwuręczne albo niejednoznaczne, wykluczone z zadania liczenia.
DROPPED = ["hand_heart", "hand_heart2", "holy", "timeout", "xsign", "take_picture", "grabbing", "grip"]

# Rozluźniona dłoń obok dłoni pokazującej gest. Nie znamy liczby palców, więc to nie jest etykieta treningowa.
NO_GESTURE = "no_gesture"

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
# Indeksy landmarków dłoni w MediaPipe (mcp, pip, dip, tip), dla kciuka (cmc, mcp, ip, tip).
FINGER_JOINTS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
