# HAL and RTOS families

`classify.py` assigns a family from the function's name. This file says what
each family's conventions are and how far they have been checked. A
convention is enough for a model only when it is marked verified below or
when the vendor header was read for the function at hand. Otherwise read the
header, read the body, or fall back to a signature-only model.

## Verified against headers in this repository

### libopencm3

Headers: `firmwares/repos/pulseox-firmware/libopencm3/include/libopencm3/`
(also under `firmwares/repos/bloodlight-firmware/firmware/libopencm3/`).
Names are lowercase with the peripheral first: `usbd_`, `usart_`, `gpio_`,
`rcc_`, `spi_`, `i2c_`, `timer_`, `flash_`, `nvic_`, `adc_`.

There is no common status type. Each function documents its own return:

| Function (`usb/usbd.h`, `stm32/common/*.h`) | Declared | Return |
| --- | --- | --- |
| `usbd_ep_write_packet(dev, addr, buf, len)` | `uint16_t` | "0 if failed, len if successful" |
| `usbd_ep_read_packet(dev, addr, buf, len)` | `uint16_t` | number of bytes read; writes `buf` |
| `usbd_ep_stall_get(dev, addr)` | `uint8_t` | stall state |
| `usbd_register_set_config_callback(dev, cb)` | `int` | "0 if successful" |
| `usbd_poll(dev)` | `void` | runs registered callbacks |
| `usart_recv(usart)`, `usart_recv_blocking(usart)` | `uint16_t` | the data register: fresh on every call |
| `usart_send_blocking(usart, data)` | `void` | none |
| `gpio_get(port, gpios)` | `uint16_t` | the masked pin state: a subset of `gpios` |

Peripheral arguments are register base addresses passed as `uint32_t`, not
pointers. A model never dereferences them.

`usbd_poll` and the other callback-running functions call back into the
firmware. A model that ignores that hides whatever the callbacks do; say so
in the report.

The lifted signature can differ from the header. The fixture
`test/patchir-klee-verifier/bl_usb__send_message.ll` declares
`i32 @usbd_ep_write_packet(ptr, ptr, i32)`, three parameters where the header
has four.

## Not verified here

No header of these families is in the repository. The notes are from general
knowledge of the libraries and can be wrong in detail. Before relying on
one, find the header in the firmware's source tree or read the function's
body.

| Family | Names | Convention to confirm |
| --- | --- | --- |
| `stm32-hal` | `HAL_<Periph>_<Action>`, `__HAL_*` macros | most functions return `HAL_StatusTypeDef`: `HAL_OK` 0, `HAL_ERROR` 1, `HAL_BUSY` 2, `HAL_TIMEOUT` 3 (`stm32xxxx_hal_def.h`). The first parameter is a handle struct the function reads and updates. `HAL_GetTick` returns a counter; `HAL_Delay` returns nothing |
| `stm32-ll` | `LL_<PERIPH>_<Action>` | thin register accessors, mostly `static inline`, so they are usually inlined and absent from the externals |
| `cmsis-core` | `NVIC_*`, `SysTick_*`, `SCB_*`, `__enable_irq`, `__get_PRIMASK` | register accessors and interrupt control; `NVIC_SystemReset` does not return |
| `cmsis-rtos` | `osThread*`, `osMutex*`, `osDelay` | v2 functions return `osStatus_t`, where 0 is `osOK` and errors are negative; creation functions return an id that is null on failure |
| `freertos` | `xTaskCreate`, `xQueueSend`, `vTaskDelay`, `pvPortMalloc` | `x` functions return `BaseType_t`, `pdPASS`/`pdTRUE` 1 and `pdFAIL`/`pdFALSE` 0; `pv` functions return a pointer that can be null; `v` functions return nothing. `xQueueReceive` writes its buffer argument |
| `zephyr` | `k_*`, `z_impl_*` | 0 on success, negative errno on failure |
| `nordic-nrfx` | `nrfx_*`, `nrf_*`, `sd_*` | `nrfx_err_t` or `ret_code_t`, with success being `NRFX_SUCCESS` / `NRF_SUCCESS` |
| `ambiq` | `am_hal_*` | `uint32_t` status, 0 is `AM_HAL_STATUS_SUCCESS` |
| `tivaware` | `GPIOPinWrite`, `UARTCharGet`, `ROM_*`, `MAP_*` | register accessors with per-function returns |
| `esp-idf` | `esp_*`, `gpio_*`, `uart_*` | `esp_err_t`, 0 is `ESP_OK` |
| `mbed-hal`, `pico-sdk` | lowercase `gpio_*`, `spi_*`, `i2c_*`, `uart_*` | per-function returns |

## Limits of the name tables

- The lowercase families (`libopencm3`, `esp-idf`, `mbed-hal`, `pico-sdk`)
  overlap. The first match in the table of `classify.py` wins, so a
  `gpio_set` from the Pico SDK is reported as libopencm3. The family is a
  hint for where to look, and the header decides.
- A firmware's own function with a HAL-like name (`gpio_init_board`) is
  classified as `hal`. With `--binary`, the address tells that it is in the
  image; read it like any internal function.
- A name that matches nothing is `unknown`, or `internal` when the binary or
  the Ghidra export has it.
