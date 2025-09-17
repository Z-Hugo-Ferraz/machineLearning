## Objetivo
O objetivo geral deste roteiro é utilizar as bibliotecas `pandas`, `numpy`, `matplotlib` e `scikit-learn`, além de uma base escolhida no [Kagle](https://www.kaggle.com/), para treinar e avaliar um algoritmo de K-Means.


## Base de Dados

A base de dados escolhida para a realização deste roteiro foi a [MBA Admission Dataset](https://www.kaggle.com/datasets/taweilo/mba-admission-dataset). Esta base possui 6194 linhas e 10 colunas, incluido uma coluna de ID da aplicação e uma coluna de status da admissão, esta é a váriavel dependente que será objeto da classificação.

### Análise da Base

A seguir foi feita uma análise do significado e composição de cada coluna presente na base com a finalidade de indentificar possíveis problemas á serem tradados posteriormente. 

=== "application_id"

    Esta coluna é composta pelos ID's das aplicações realizadas, ou seja trata-se de um valor numérico lógico, único a cada aplicação, desta forma pode-se afirmar que esta coluna não terá relevância para o algoritmo e deverá ser retirada da base para treinamento.

    ```python exec="on" html="1"
    --8<-- "docs/base/colunas/id.py"
    ```

=== "gender"

    Esta coluna é preenchida com o genêro do aplicante, contendo apenas valores textuais entre *"male"* e *"female"*, não incluindo opções como *"non-binary"*, *"other"* ou *"prefer not to inform"*. Logo, estes dados, por serem textuais e apresentarem binariedade, deverão ser transformados em uma variável binária numérica para que se atinja um melhor desempenho do algoritmo.

    ```python exec="on" html="1"
    --8<-- "docs/base/colunas/gender.py"
    ```

=== "international"

    Esta coluna é preenchida com valores booleanos que classificam o aplicantente como *"estrangeiro"* ou *"não-estrangeiro"*. Logo, estes dados, por serem textuais e apresentarem binariedade, deveriam ser transformados em uma variável binária numérica para que se atinja um melhor desempenho do algoritmo.

    Entretanto, a classificação desta coluna tambem poder ser notada na coluna *"race"*, pois todos os valores nulos presentes na posterior são unicamente referentes a alunos estrangeiros.

    ```python exec="on" html="1"
    --8<-- "docs/base/colunas/international.py"
    ```

=== "gpa"

    Esta coluna representa a performance acadêmica prévia do aplicante, que é calculada a partir do histórico escolar. Neste as notas particulares de cada matéria podem variar de 0 á 4, 0 sendo a pior nota possível e 4 a maior. Neste caso os GPA's dos aplicantes variam entre 2.65 e 3.77, apresentando uma curva normal. Devido ao fato destes valores serem numéricos e a maioria das variáveis do modelo serem binárias ou *dummies*, esta deve ser padronizada para valores entre 0 e 1.

    ```python exec="on" html="1"
    --8<-- "docs/base/colunas/gpa.py"
    ```

=== "major"

    Esta coluna representa em que curso o aplicante deseja entrar, podendo assumir um de três valores textuais: *"Humanities"*, *"STEM"* e *"Business"*. Neste caso, como a variavel é textual, não apresenta binariedade e não possui noção de escala (como em "ruim", "regular" e "bom"), a técnica correta para o tratamento desta coluna será o *"One Hot"*, transformando-a em 2 variáveis *dummies*.

    ```python exec="on" html="1"
    --8<-- "docs/base/colunas/major.py"
    ```

=== "race"

    Esta coluna representa a indentificação racial do aplicante, porém tambem há diversas linhas com valor nulo nesta coluna. Ao comparar o preenchimento desta coluna com as demais, percebe-se que o valor desta coluna so se apresenta nulo para estudantes estrangeiros, tornando a coluna *"international"* redundante.

    Desta forma, para otimizar o modelo, devemos remover a coluna *"international"*, prezando pela menor quantidade de colunas possível, e gerar *dummies* para cada valor registrado na coluna, pois esta não possui noção de escala (como em "ruim", "regular" e "bom"). 

    ```python exec="on" html="1"
    --8<-- "docs/base/colunas/race.py"
    ```

=== "gmat"

    Esta coluna representa o desempenho do aplicante na prova de adimissão, variando de 570 á 780, porém estas notas não apresentam uma curva normal, pois há muitos registros de notas menores que a média a mais do que há registos de notas maiores que a média. Devido ao fato destes valores serem numéricos e a maioria das variáveis do modelo serem binárias ou *dummies*, esta deve ser padronizada para valores entre 0 e 1.

    ```python exec="on" html="1"
    --8<-- "docs/base/colunas/gmat.py"
    ```

=== "work_exp"

    Esta coluna representa o tempo de experiência prévia do aplicante no mercado, exibida em anos. Os valores podem variar de 1 á 9, apresentando uma curva normal. Devido ao fato destes valores serem numéricos e a maioria das variáveis do modelo serem binárias ou *dummies*, esta deve ser padronizada para valores entre 0 e 1.

    ```python exec="on" html="1"
    --8<-- "docs/base/colunas/work.py"
    ```

=== "work_industry"

    Esta coluna representa a área de experiência prévia do aplicante no mercado, podendo assumir, nesta base um de quatorze valores textuais. E como esta coluna não apresenta binariedade e não possui noção de escala (como em "ruim", "regular" e "bom"), a técnica correta para o tratamento desta coluna será o *"One Hot"*, transformando-a em 13 variáveis *dummies*.

    ```python exec="on" html="1"
    --8<-- "docs/base/colunas/workInd.py"
    ```

=== "admission"

    Esta coluna apresenta valores em texto para os aplicantes admitos e na lista de espera, além de valores nulos para aqueles que não foram aceitos. Esta coluna é o objeto da classificação e portanto será separada das outras colunas da base, e os valores nulos deveram ser preenchidos.

    ```python exec="on" html="1"
    --8<-- "docs/base/colunas/admission.py"
    ```

## Pré-processamento

Esta secção visa preparar os dados para o treinamento da árvore de decisão, atendendo as observações e análises feitas no tópico anterior.

=== "Base preparada"
    ```python exec="1"
    --8<-- "docs/knn/prepair.py"
    ```
=== "code"
    ```python exec="0"
    --8<-- "docs/knn/prepair.py"
    ```
=== "Base original"
    ```python exec="1"
    --8<-- "docs/base/baseViz.py"
    ```

## Divisão dos dados 

Devido a composição da coluna de *admission*, a seperação dos dados deve ser feita com maior atenção. Caso esta separação fosse feita com aleatoriedade, haveria a possibilidade de que a base de treinamento tornar-se enviesada. Portanto, esta deve ser executada com proporcionalidade a composição da coluna alvo. Tendo em vista situações como esta o `sickit-learn` já implementou o sorteamento extratificado como a opção `stratify` no comando `train_test_split()`.

Além disto para o treinamento foi utilizado uma separação arbitrária da base em 70% treinamento e 30% validação.


```python exec="0"
--8<-- "docs/knn/separar.py"
```

## Treinamento do Modelo

=== "Treinamento"
    ```python exec="1"
    --8<-- "docs/kmeans/train.py"
    ```
=== "code"
    ```python exec="0"
    --8<-- "docs/kmeans/train.py"
    ```