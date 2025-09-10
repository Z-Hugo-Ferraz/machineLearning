## Objetivo
O objetivo geral deste roteiro é utilizar as bibliotecas `pandas`, `numpy`, `matplotlib` e `scikit-learn`, além de uma base escolhida no [Kagle](https://www.kaggle.com/), para treinar e avaliar um algoritmo de árvore de decisão.


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

    Esta coluna é preenchida com o genêro do aplicante, contendo apenas valores textuais entre *"male"* e *"female"*, não incluindo opções como *"non-binary"*, *"other"* ou *"prefer not to inform"*. Logo, estes dados, por serem textuais e apresentarem binariedade, deverão ser transformados em uma variável *dummy* para que se atinja um melhor desempenho do algoritmo.

    ```python exec="on" html="1"
    --8<-- "docs/base/colunas/gender.py"
    ```

=== "international"

    Esta coluna é preenchida com valores booleanos que classificam o aplicantente como *"estrangeiro"* ou *"não-estrangeiro"*. Logo, estes dados, por serem textuais e apresentarem binariedade, deveriam ser transformados em uma variável *dummy* para que se atinja um melhor desempenho do algoritmo.

    Entretanto, a classificação desta coluna tambem poder ser notada na coluna *"race"*, pois todos os valores nulos presentes na posterior são unicamente referentes a alunos estrangeiros.

    ```python exec="on" html="1"
    --8<-- "docs/base/colunas/international.py"
    ```

=== "gpa"

    Esta coluna representa a performance acadêmica prévia do aplicante, que é calculada a partir do histórico escolar. Neste as notas particulares de cada matéria podem variar de 0 á 4, 0 sendo a pior nota possível e 4 a maior. Neste caso os GPA's dos aplicantes variam entre 2.65 e 3.77, apresentando uma curva normal. Devido ao fato destes valores já serem numéricos estes já estão adequados para o modelo.

    ```python exec="on" html="1"
    --8<-- "docs/base/colunas/gpa.py"
    ```

=== "major"

    Esta coluna representa em que curso o aplicante deseja entrar, podendo assumir um de três valores textuais: *"Humanities"*, *"STEM"* e *"Business"*. Neste caso, como a variavel é textual e não apresenta binariedade, a técnica correta para o tratamento desta coluna será o *Label Enconding*, transformando estes valores textuais em valores númericos.

    ```python exec="on" html="1"
    --8<-- "docs/base/colunas/major.py"
    ```

=== "race"

    Esta coluna representa a indentificação racial do aplicante, porém tambem há diversas linhas com valor nulo nesta coluna. Ao comparar o preenchimento desta coluna com as demais, percebe-se que o valor desta coluna so se apresenta nulo para estudantes estrangeiros, tornando a coluna *"international"* redundante.

    Desta forma, para otimizar o modelo, devemos remover a coluna *"international"*, prezando pela menor quantidade de colunas possível. E como esta coluna não apresentar binariedade, deverá ser utilizada a técnica de *Label Enconding*, transformando estes valores textuais e nulos em valores númericos. 

    ```python exec="on" html="1"
    --8<-- "docs/base/colunas/race.py"
    ```

=== "gmat"

    Esta coluna representa o desempenho do aplicante na prova de adimissão, variando de 570 á 780, porém estas notas não apresentam uma curva normal, pois há muitos registros de notas menores que a média a mais do que há registos de notas maiores que a média. Devido ao fato destes valores já serem numéricos estes já estão adequados para o modelo.

    ```python exec="on" html="1"
    --8<-- "docs/base/colunas/gmat.py"
    ```

=== "work_exp"

    Esta coluna representa o tempo de experiência prévia do aplicante no mercado, exibida em anos. Os valores podem variar de 1 á 9, apresentando uma curva normal. Devido ao fato destes valores já serem numéricos estes já estão adequados para o modelo.

    ```python exec="on" html="1"
    --8<-- "docs/base/colunas/work.py"
    ```

=== "work_industry"

    Esta coluna representa a área de experiência prévia do aplicante no mercado, podendo assumir, nesta base um de quatorze valores textuais. E como esta coluna não apresenta binariedade, deverá ser utilizada a técnica de *Label Enconding*, transformando estes valores textuais em valores númericos.

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
    --8<-- "docs/arvore/prepair.py"
    ```
=== "code"
    ```python exec="0"
    --8<-- "docs/arvore/prepair.py"
    ```
=== "Base original"
    ```python exec="1"
    --8<-- "docs/base/baseViz.py"
    ```

## Divisão dos dados 

Devido a composição da coluna de *admission*, a seperação dos dados deve ser feita com maior atenção. Caso esta separação fosse feita com aleatoriedade, haveria a possibilidade de que a base de treinamento tornar-se enviesada. Portanto, esta deve ser executada com proporcionalidade a composição da coluna alvo. Tendo em vista situações como esta o `sickit-learn` já implementou o sorteamento extratificado como a opção `stratify` no comando `train_test_split()`.

Além disto para o treinamento foi utilizado uma separação arbitrária da base em 70% treinamento e 30% validação.


```python exec="0"
--8<-- "docs/arvore/separar.py"
```

## Treinamento da Árvore

=== "Modelo da Árvore"
    ```python exec="on" html="1"    
    --8<-- "docs/arvore/train.py"
    ```
=== "code"
    ```python exec="0"    
    --8<-- "docs/arvore/train.py"
    ```

## Avaliação do Modelo

Com este treinamento o modelo apresenta 77.78% de precisão, número satisfatório para um modelo de classificação real, e as colunas mais importantes em sua tomada de deicisão são as ponutações *gpa* e *gmat* com 31.2% e 29.1% de importância, respectivamente, e a coluna com menor relevancia para o modelo é a *gender*, com  1.6% de importância.

Entretando utilizar mais dados no treinamento do modelo poderia melhorara sua precisão. Logo, para compravar esta hipótese o modelo será treinado novamente com 80% da base de dados original para treinamento.

## Retreinamento

=== "Modelo da Árvore"
    ```python exec="on" html="1"    
    --8<-- "docs/arvore/train2.py"
    ```
=== "code"
    ```python exec="0"    
    --8<-- "docs/arvore/train2.py"
    ```

## Avaliação do novo modelo

Com este retreinamento a hipótese anterior é rejeitada, pois ao utilizar 80% da base para treinamento a precisão geral do modelo caiu para 77.16%. Entretanto, as métricas de *gpa* e *gmat* continuaram sendo as mais relevantes, comprovando sua importância para o modelo.

## Conclusão

Ao fim deste roteiro nota-se que as colunas não precisam estar normalizadas para que se treine uma árvore de decisão, aumentar os dados de treinamento do modelo, em detrimento dos dados de teste, pode prejudicar a precisão geral do mesmo e que grande parte do tempo de trabalho do cientista de dados é a análise e limpeza da base de dados original.  
