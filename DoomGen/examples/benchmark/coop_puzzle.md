TODO: Consider using SectorAction:
```
https://zdoom.org/wiki/Classes:SecActEnter
https://zdoom.org/wiki/Classes:SecActExit
https://zdoom.org/wiki/Classes:SecActHitFloor

https://zdoom.org/wiki/Classes:SectorAction
```

```acs
#include "zcommon.acs"

global int 1:zone;
global int 2:plate_state;
global int 3:room_type; // Unused

int plate2_timer[50];

int active_plate[51];

// 0 = Agent A, 1 = Agent B
int plate_in_b_zone[51];

// For each zone from zone i from 1 to zone 50
// zone sectors are tagged from 201 to 250
// Plate 1 (single press) in agent A's zone is tagged from i*1000+1 to i*1000+99
// Plate 1 (single press) in agent B's zone is tagged from i*1000+101 to i*1000+199
// Zones are paired such as (i, i+1), i is odd and i+1 is hallway, and they share the same door pair
// Agent A's doors are tagged from 1-50, incrementally
// Agent B's doors are tagged from 51-100, incrementally
// Plate 2 (both press) is tagged 101-150 and opens door 151-200

// Zone: 201-250
// Door A type 1: 1-50
// Door B type 1: 51-100
// Plate A type 1: i*1000+1 to i*1000+99
// Plate B type 1: i*1000+101 to i*1000+199
// Plate type 2: 101-150
// Door type 2: 151-200

function int cnt_sectors(int min_tag, int max_tag) {
	int count = 0;
	int t;
	for (t = min_tag; t <= max_tag; t++) {
		if (GetSectorLightLevel(t) != -1) {
			count++;
		}
	}
	return count;
}

function int get_sector_tag(int min_tag, int max_tag, int n) {
	int count = 0;
	int t;
	for (t = min_tag; t <= max_tag; t++) {
		if (GetSectorLightLevel(t) != -1) {
			count++;
			if (count == n) {
				return t;
			}
		}
	}
	return -1;
}

script 2 OPEN {
	int i;
	int pos;
	int iidx;
	int cnt_a;
	int cnt_b;
	int chosen;
	int rand; // 0 = A, 1 = B

	rand = Random(0, 1);
	for (i = 1; i <= 50; i++) {
        if (i % 2 != 0) {
            rand = Random(0, 1);
        }
		iidx = i * 1000;

		cnt_a = cnt_sectors(iidx + 1, iidx + 99);
		cnt_b = cnt_sectors(iidx + 101, iidx + 199);

		if (cnt_a + cnt_b == 0) {
			active_plate[i] = 0;
			continue;
		}

		if (rand == 0 && cnt_a > 0) {
			pos = Random(1, cnt_a);
			chosen = get_sector_tag(iidx + 1, iidx + 99, pos);
			active_plate[i] = chosen;
			plate_in_b_zone[i] = 0;
			ChangeFloor(chosen, "AQF044"); // Agent A plate
		} else if (rand == 1 && cnt_b > 0) {
			pos = Random(1, cnt_b);
			chosen = get_sector_tag(iidx + 101, iidx + 199, pos);
			active_plate[i] = chosen;
			plate_in_b_zone[i] = 1;
			ChangeFloor(chosen, "AQF038"); // Agent B plate
		} else if (cnt_a > 0) {
			pos = Random(1, cnt_a);
			chosen = get_sector_tag(iidx + 1, iidx + 99, pos);
			active_plate[i] = chosen;
			plate_in_b_zone[i] = 0;
			ChangeFloor(chosen, "AQF044"); // Agent A plate
		} else {
			pos = Random(1, cnt_b);
			chosen = get_sector_tag(iidx + 101, iidx + 199, pos);
			active_plate[i] = chosen;
			plate_in_b_zone[i] = 1;
			ChangeFloor(chosen, "AQF038"); // Agent B plate
		}

		rand = 1 - rand;
	}
}

script 1 OPEN {
	int new_zone;
	int plate;
	int z;
	int i;
	int d;
	int idx;
	int p1_on;
	int p2_on;
	int should_open;
	int door_tag;
	int plate_tag;

	Delay(5);

	// Player 1 = TID 1, Player 2 = TID 2
	SetActivator(0, AAPTR_PLAYER1);
	Thing_ChangeTID(0, 1);
	SetActivator(0, AAPTR_PLAYER2);
	Thing_ChangeTID(0, 2);
	SetActivator(0, AAPTR_NULL);

	while (true) {
		new_zone = 0;
		plate = 0;

		// Zone sectors
		for (z = 201; z <= 250; z++) {
			if (ThingCountSector(T_NONE, 1, z) > 0 || ThingCountSector(T_NONE, 2, z) > 0) {
				new_zone = z - 200;
			}
		}

		// Plate type 1
		for (i = 1; i <= 50; i++) {
			plate_tag = active_plate[i];
			if (plate_tag == 0) continue;

			p1_on = ThingCountSector(T_NONE, 1, plate_tag) > 0;
			p2_on = ThingCountSector(T_NONE, 2, plate_tag) > 0;

			if (p1_on || p2_on) {
				plate = 1;
				if (new_zone == 0) {
					new_zone = i;
				}
			}

			int door = (i + 1) / 2;
			if (plate_in_b_zone[i] == 1) {
				door_tag = door;
			} else {
				door_tag = door + 50;
			}

			should_open = (p1_on || p2_on);
			if (should_open) {
				Door_Open(door_tag, 64, 0);
			} else {
				Door_Close(door_tag, 64, 0);
			}
		}

		// Plate type 2
		for (d = 101; d <= 150; d++) {
			idx = d - 101;

			p1_on = ThingCountSector(T_NONE, 1, d) > 0;
			p2_on = ThingCountSector(T_NONE, 2, d) > 0;

			if (p1_on && p2_on) {
				plate = 1;
				plate2_timer[idx] = 70; // 2s
				if (new_zone == 0) new_zone = d - 100;
			}

			should_open = (p1_on && p2_on) || (plate2_timer[idx] > 0);
			if (should_open) {
				Door_Open(d + 50, 64, 0);
			} else {
				Door_Close(d + 50, 64, 0);
			}

			if (plate2_timer[idx] > 0) plate2_timer[idx]--;
		}

		if (new_zone > 0) zone = new_zone;
		plate_state = plate;

		Delay(1);
	}
}
```
