/* Minimal preemptive RTOS kernel implementation (Cortex-M3). */
#include "rtos_kernel.h"
#include "stm32f1xx.h"

static os_tcb_t os_tcbs[OS_MAX_TASKS];
os_tcb_t *volatile os_current = 0;
volatile uint8_t os_started = 0;
static volatile uint32_t os_tick_count = 0;

#define OS_STACK_MAGIC 0xDEADBEEFu

/* 诊断打点：本工程不提供 trace 模块时用空实现（weak），
   若外部（如 fault_dump.c）提供了强定义则自动覆盖。 */
#pragma weak diag_mark
void diag_mark(unsigned stage) { (void)stage; }

void os_init(void)
{
    for (int i = 0; i < OS_MAX_TASKS; i++) {
        os_tcbs[i].used = 0;
        os_tcbs[i].blocked = 0;
        os_tcbs[i].delay = 0;
        os_tcbs[i].sp = 0;
    }
    os_current = 0;
    os_started = 0;
    os_tick_count = 0;
}

int os_task_create(os_task_fn fn, void *arg, uint32_t *stack,
                   uint32_t stack_words, const char *name, uint8_t prio)
{
    int idx = -1;
    for (int i = 0; i < OS_MAX_TASKS; i++) if (!os_tcbs[i].used) { idx = i; break; }
    if (idx < 0) return -1;

    os_tcb_t *t = &os_tcbs[idx];
    t->used = 1; t->started = 0; t->blocked = 0; t->delay = 0;
    t->fn = fn; t->arg = arg; t->name = name; t->prio = prio;
    t->stack = stack; t->stack_words = stack_words;

    for (uint32_t i = 0; i < stack_words; i++) stack[i] = OS_STACK_MAGIC;

    /* build initial exception frame: r4-r11 then r0,r1,r2,r3,r12,lr,pc,xpsr */
    uint32_t *sp = stack + stack_words;
    *(--sp) = 0x01000000u;          /* xPSR */
    *(--sp) = (uint32_t)fn;         /* PC */
    *(--sp) = 0xFFFFFFFDu;          /* LR (EXC_RETURN: thread mode, PSP) */
    *(--sp) = 0;                    /* R12 */
    *(--sp) = 0;                    /* R3  */
    *(--sp) = 0;                    /* R2  */
    *(--sp) = 0;                    /* R1  */
    *(--sp) = (uint32_t)arg;        /* R0  */
    for (int i = 0; i < 8; i++) *(--sp) = 0;   /* R4-R11 */
    t->sp = sp;
    return idx;
}

static os_tcb_t *os_pick_next(void)
{
    int start = 0;
    if (os_current) {
        for (int i = 0; i < OS_MAX_TASKS; i++) if (&os_tcbs[i] == os_current) { start = i + 1; break; }
    }
    for (int pass = 0; pass < OS_MAX_TASKS; pass++) {
        int best = -1; uint8_t best_prio = 0xFF;
        for (int k = 0; k < OS_MAX_TASKS; k++) {
            int i = (start + pass + k) % OS_MAX_TASKS;
            os_tcb_t *t = &os_tcbs[i];
            if (!t->used || t->delay > 0 || t->blocked) continue;
            if (t->prio < best_prio) { best_prio = t->prio; best = i; }
        }
        if (best >= 0) return &os_tcbs[best];
    }
    return os_current;
}

os_tcb_t *os_switch_c(void)
{
    os_tcb_t *next = os_pick_next();
    os_current = next;
    os_started = 1;
    return next;
}

__attribute__((naked)) void os_pendsv_asm(void)
{
    __asm volatile(
        "mrs    r0, psp                 \n"
        "ldr    r3, =os_started         \n"
        "ldrb   r3, [r3]                \n"
        "cbz    r3, 1f                  \n"
        "stmdb  r0!, {r4-r11}           \n"
        "ldr    r1, =os_current         \n"
        "ldr    r2, [r1]                \n"
        "str    r0, [r2]                \n"
        "1:                             \n"
        "bl     os_switch_c             \n"
        "ldr    r1, [r0]                \n"
        "ldmia  r1!, {r4-r11}           \n"
        "msr    psp, r1                 \n"
        "ldr    lr, =0xFFFFFFFD         \n"
        "bx     lr                      \n"
    );
}

void os_delay(uint32_t ms)
{
    __disable_irq();
    os_current->delay = (ms == 0) ? 1 : ms;
    __enable_irq();
    SCB->ICSR = SCB_ICSR_PENDSVSET_Msk;
    __asm volatile("dsb\n isb\n");
}

uint32_t os_ticks(void) { return os_tick_count; }

uint32_t os_stack_watermark(uint8_t idx)
{
    if (idx >= OS_MAX_TASKS || !os_tcbs[idx].used) return 0;
    os_tcb_t *t = &os_tcbs[idx];
    uint32_t n = 0;
    while (n < t->stack_words && t->stack[n] == OS_STACK_MAGIC) n++;
    return t->stack_words - n;
}

void os_sem_init(os_sem_t *s, int32_t initial) { s->count = initial; }

void os_sem_post(os_sem_t *s)
{
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    s->count++;
    for (int i = 0; i < OS_MAX_TASKS; i++)
        if (os_tcbs[i].used && os_tcbs[i].blocked) os_tcbs[i].blocked = 0;
    SCB->ICSR = SCB_ICSR_PENDSVSET_Msk;
    if (!primask) __enable_irq();     /* 只有调用前是开中断状态才恢复 */
}

void os_sem_wait(os_sem_t *s)
{
    for (;;) {
        uint32_t primask = __get_PRIMASK();
        __disable_irq();
        if (s->count > 0) { s->count--; if (!primask) __enable_irq(); return; }
        if (os_current) os_current->blocked = 1;
        SCB->ICSR = SCB_ICSR_PENDSVSET_Msk;
        if (!primask) __enable_irq();
        __asm volatile("dsb\n isb\n");
    }
}

void os_tick_isr(void)
{
    os_tick_count++;
    int need_switch = 0;
    for (int i = 0; i < OS_MAX_TASKS; i++) {
        os_tcb_t *t = &os_tcbs[i];
        if (!t->used) continue;
        if (t->delay > 0) { if (--t->delay == 0) need_switch = 1; }
    }
    if (need_switch) SCB->ICSR = SCB_ICSR_PENDSVSET_Msk;
}

static void os_tick_timer_init(void)
{
    RCC->APB1ENR |= RCC_APB1ENR_TIM7EN;
    TIM7->PSC = 7200 - 1;
    TIM7->ARR = 10 - 1;
    TIM7->EGR = TIM_EGR_UG;
    TIM7->SR  = 0;
    TIM7->DIER |= TIM_DIER_UIE;
    NVIC_SetPriority(TIM7_IRQn, 15);
    NVIC_EnableIRQ(TIM7_IRQn);
    TIM7->CR1 |= TIM_CR1_CEN;
}

void TIM7_IRQHandler(void)
{
    if (TIM7->SR & TIM_SR_UIF) { TIM7->SR &= ~TIM_SR_UIF; os_tick_isr(); }
}

void os_start(void)
{
    { extern void diag_mark(unsigned); diag_mark(7); }
    os_current = os_pick_next();
    if (!os_current) { while (1) {} }
    os_tick_timer_init();
    NVIC_SetPriority(PendSV_IRQn, 15);
    __set_PSP((uint32_t)os_current->sp);
    __set_CONTROL(0x02);
    __ISB();
    __enable_irq();
    SCB->ICSR = SCB_ICSR_PENDSVSET_Msk;   /* first switch: restore only */
    __asm volatile("dsb\n isb\n");
    while (1) {}
}
