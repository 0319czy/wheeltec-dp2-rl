/* Lightweight RTOS task set for the first-order RL firmware.
 * Real-time control (MPU6050 EXTI, 100 Hz) stays in the interrupt on purpose:
 * earlier attempts to move it into a task broke timing on this board, and the
 * RL policy needs a jitter-free 10 ms cadence. The RTOS hosts everything that
 * is periodic but not hard-real-time: telemetry/reporting, LiDAR servicing and
 * kernel housekeeping.
 * Comments kept ASCII on purpose (previous GBK/UTF-8 mixing broke anchors).
 */
#include "rtos_kernel.h"
#include "main.h"
#include <stdio.h>

extern void DP2_FirstOrderReport(void);
extern void Lidar_Update(void);
extern void Lidar_SendPointCloud(void);
extern void Lidar_ServiceFastPointCloud(void);

static uint32_t stack_report[512];
static uint32_t stack_lidar[640];
static uint32_t stack_mon[256];

volatile uint32_t g_report_loops = 0;
volatile uint32_t g_lidar_loops = 0;
volatile uint32_t g_mon_ticks = 0;

/* task 2: telemetry / report frames (was the main loop body) */
static void task_report(void *arg)
{
    (void)arg;
    for (;;) {
        DP2_FirstOrderReport();
        g_report_loops++;
        os_delay(1);
    }
}

/* task 3: LiDAR servicing: update, fast path first, then full point cloud */
static void task_lidar(void *arg)
{
    (void)arg;
    for (;;) {
        Lidar_Update();
        Lidar_ServiceFastPointCloud();
        Lidar_SendPointCloud();
        g_lidar_loops++;
        os_delay(2);
    }
}

/* task 4: housekeeping / stack watermark probe (read via debugger or later frame) */
static void task_mon(void *arg)
{
    (void)arg;
    for (;;) {
        g_mon_ticks++;
        os_delay(1000);
    }
}

void rtos_start_tasks(void)
{
    os_init();
    os_task_create(task_report, 0, stack_report, 512, "report", 2);
    os_task_create(task_lidar,  0, stack_lidar,  640, "lidar",  3);
    os_task_create(task_mon,    0, stack_mon,    256, "mon",    4);
    os_start();
}
