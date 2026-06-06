import time

from sound_beep import SoundBeep, TargetDistance

beeper = SoundBeep()

beeper.start()

time.sleep(5)

beeper.set_target_distance(TargetDistance.DETECTED)
time.sleep(5)

beeper.set_target_distance(TargetDistance.CLOSE)
time.sleep(2)

beeper.stop()
