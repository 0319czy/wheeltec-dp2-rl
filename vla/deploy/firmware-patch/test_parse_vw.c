/* test_parse_vw.c - PC unit test for deploy/firmware-patch/vla_cmd.h.
 *
 * Build and run (no STM32 toolchain needed):
 *     gcc -std=c99 -Wall -Wextra -I deploy/firmware-patch \
 *         deploy/firmware-patch/test_parse_vw.c -lm -o /tmp/test_parse_vw
 *     /tmp/test_parse_vw
 *
 * It checks the exact code that the firmware patch adds:
 *   1. legal #V<v_mm_s>,<w_mrad_s>; frames parse (with '#'/' ;' stripped by caller)
 *   2. clamping to spec limits (+/-350 mm/s, +/-1000 mrad/s)
 *   3. malformed frames are rejected
 *   4. delta-theta conversion matches the Normal() 10 ms/0.05 rad equivalence
 *   5. 300 ms dead-man logic, including uint32 tick wrap-around
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "vla_cmd.h"

static int failures = 0;
static int checks = 0;

static void check(int condition, const char *what)
{
    checks++;
    if (condition) {
        printf("[ OK ] %s\n", what);
    } else {
        failures++;
        printf("[FAIL] %s\n", what);
    }
}

static int near_value(float got, float expected, float tol)
{
    return fabsf(got - expected) <= tol;
}

static void check_parse(const char *body, int expected_ok,
                        int expected_v, int expected_w)
{
    int16_t v = 0;
    int16_t w = 0;
    int ok = vla_parse_vw(body, &v, &w);
    char label[160];
    if (expected_ok) {
        snprintf(label, sizeof(label), "parse %-14s -> v=%d w=%d", body, v, w);
        check(ok == 1 && v == expected_v && w == expected_w, label);
    } else {
        snprintf(label, sizeof(label), "reject %-13s (ok=%d)", body, ok);
        check(ok == 0, label);
    }
}

int main(void)
{
    float dl = 0.0f;
    float dr = 0.0f;
    int16_t v = 0;
    int16_t w = 0;

    printf("== vla_cmd.h unit test ==\n");

    /* 1. legal frames */
    check_parse("V250,0", 1, 250, 0);
    check_parse("V-200,0", 1, -200, 0);
    check_parse("V150,800", 1, 150, 800);
    check_parse("V0,-1000", 1, 0, -1000);
    check_parse("V0,0", 1, 0, 0);

    /* 2. clamping */
    check_parse("V9999,5000", 1, 350, 1000);
    check_parse("V-9999,-5000", 1, -350, -1000);

    /* 3. malformed frames must be rejected */
    check_parse("", 0, 0, 0);
    check_parse("V", 0, 0, 0);
    check_parse("V250", 0, 0, 0);
    check_parse("V250,", 0, 0, 0);
    check_parse("V,0", 0, 0, 0);
    check_parse("V250,0x", 0, 0, 0);
    check_parse("V250,0,1", 0, 0, 0);
    check_parse("#V250,0", 0, 0, 0);
    check_parse("v250,0", 0, 0, 0);
    check_parse("V+250,0", 0, 0, 0);
    check_parse("V250,0 ", 0, 0, 0);
    check_parse("V250.0,0", 0, 0, 0);
    check_parse("V1234567,0", 0, 0, 0);

    /* 4. delta theta: forward 0.25 m/s -> 0.25*0.01/0.0335 rad per tick */
    check(vla_parse_vw("V250,0", &v, &w) == 1, "parse V250,0 for delta");
    vla_delta_theta(v, w, &dl, &dr);
    check(near_value(dl, 0.25f * 0.01f / 0.0335f, 1e-5f) &&
          near_value(dr, 0.25f * 0.01f / 0.0335f, 1e-5f),
          "delta forward: dl == dr == v*dt/r");
    check(near_value(dl, 0.074627f, 1e-4f), "delta forward numeric 0.074627 rad");

    /* 5. delta theta: backward + right turn, hand-computed */
    check(vla_parse_vw("V-200,-800", &v, &w) == 1, "parse V-200,-800 for delta");
    vla_delta_theta(v, w, &dl, &dr);
    check(near_value(dl, -0.037015f, 1e-4f) && near_value(dr, -0.082388f, 1e-4f),
          "delta backward+right: dl=-0.037015 dr=-0.082388 rad");

    /* 6. delta theta: pure left spin -> right wheel +, left wheel - */
    check(vla_parse_vw("V0,800", &v, &w) == 1, "parse V0,800 for delta");
    vla_delta_theta(v, w, &dl, &dr);
    check(near_value(dl, -0.022687f, 1e-4f) && near_value(dr, 0.022687f, 1e-4f),
          "delta spin left: dl<0, dr>0, symmetric");

    /* 7. existing char-command equivalence: 0.05 rad / 10 ms ~= 0.1675 m/s */
    check(vla_parse_vw("V168,0", &v, &w) == 1, "parse V168,0 for legacy equivalence");
    vla_delta_theta(v, w, &dl, &dr);
    check(near_value(dl, 0.05f, 1.5e-3f) && near_value(dr, 0.05f, 1.5e-3f),
          "legacy forward ~= movement_speed 0.05 rad/tick");

    /* 8. 300 ms dead-man, normal and wrapped tick counter */
    check(vla_fresh(1000u, 1000u) == 1, "watchdog fresh at dt=0 ms");
    check(vla_fresh(1300u, 1000u) == 1, "watchdog fresh at dt=300 ms");
    check(vla_fresh(1301u, 1000u) == 0, "watchdog stale at dt=301 ms");
    check(vla_fresh(5u, 0xFFFFFFF0u) == 1, "watchdog wrap-around 32-bit tick");

    printf("== %d checks, %d failed ==\n", checks, failures);
    if (failures == 0) {
        printf("PASSED\n");
        return 0;
    }
    printf("FAILED\n");
    return 1;
}
