# Cursachiko
## Быстрое преобразование Фурье и его распараллеливание (под архитектуру CUDA для NVIDIA)
Содержание работы:
1. Реализация вычисления БПФ без распараллеливания.
2. Реализация вычисления БПФ с распараллеливанием на CPU.
3. Реализация вычисления БПФ с распараллеливанием на CUDA.
4. Сравнение разных подходов к вычислению преобразования Фурье:
  * на малых объемах данных (помещающихся в кэш);
  * на средних объемах данных (помещающихся в GPU);
  * на больших объемах данных (не помещающихся в GPU);
