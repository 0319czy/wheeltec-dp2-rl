/* Minimal preemptive RTOS kernel for Cortex-M3 (STM32F103).
   Features: fixed-priority preemption, TIM7 1ms tick, PendSV context switch,
   static task stacks, counting semaphores. Comments kept ASCII on purpose. */
#ifndef RTOS_KERNEL_H
#define RTOS_KERNEL_H

#include <stdint.h>

#define OS_MAX_TASKS   6

typedef void (*os_task_fn)(void *arg);

typedef struct {
    uint32_t *sp;            /* saved stack pointer (context switch) */
    uint32_t *stack;
    uint32_t  stack_words;
    os_task_fn fn;
    void     *arg;
    const char *name;
    volatile uint32_t delay; /* remaining ticks; 0 = ready */
    uint8_t   prio;
    uint8_t   used;
    uint8_t   started;
    volatile uint8_t blocked;
} os_tcb_t;

/* counting semaphore */
typedef struct { volatile int32_t count; } os_sem_t;

void os_init(void);
int  os_task_create(os_task_fn fn, void *arg, uint32_t *stack,
                    uint32_t stack_words, const char *name, uint8_t prio);
void os_start(void);
void os_delay(uint32_t ms);
uint32_t os_ticks(void);
uint32_t os_stack_watermark(uint8_t idx);

void os_sem_init(os_sem_t *s, int32_t initial);
void os_sem_post(os_sem_t *s);   /* ISR safe */
void os_sem_wait(os_sem_t *s);   /* blocks the calling task */

void os_tick_isr(void);          /* called from TIM7 interrupt (1 ms) */

#endif
