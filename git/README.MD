# gitcontribution
This script shows the number of lines in the current branch by authors. If the code was added and deleted in the past, the script will not include those deleted code.

Example: `git ls-files -- ':(exclude)*.html' ':(exclude)*.pdf' ':(exclude)*.png' | gitcontribution`

Example output:
```
Blaming file: src/game/items/Scimitar.java
Blaming file: src/game/items/Tradeable.java
.
.
.
Blaming file: src/game/items/Uchigatana.java
    169 Author1
    1885 Author2
    40 Author3
```

# gitstat
This script shows the number of lines added and removed by authors during the lifetime of the entire repo.

Example1: `gitstat e` => this shows the contribution breakdown by email.

Example1 output:
```
author.1@example.com +92 -0
author.2@example.com +847 -247
author.3@example.com +1322 -621
```

Example2: `gitstat n` => this shows the contribution breakdown by author name.

Example2 output:
```
author.1 +92 -0
author.2 +847 -247
author.3 +1322 -621
```