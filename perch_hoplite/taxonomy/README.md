# Taxonomy

This is a micro-library for managing labels from various domains, and
transformations between different sets of labels.

There are a variety of problems which arise in label management. Some of these
problems are intermittent for a given domain, but are exacerbated in areas where
there are many (thousands) of labels. For example:

*   Multiple taxonomies: When multiple taxonomies exist in a given area, we will
    want methods for easy conversion between the taxonomies.
*   Taxonomic updates: When revisions to taxonomies are published, we need to
    update label sets in a consistent way.
*   Non-standard dataset annotations: Often datasets are annotated with ad hoc
    labels, chosen by the researchers. We want to convert these labels to a
    standard taxonomy.
*   Specific label sets: We often want to handle a specific set of labels, such
    as when training classifiers for different subsets of bird species.

By providing a dedicated library for label handling, we hope to isolate solving
these problems from other code.

## Namespaces, Mappings, and ClassLists.

We provide three high-level objects for managing labels:

*   A **namespace** is a fixed (unordered) set of labels. Think of these as a
    *universe* of labels. Some universes are big (e.g., all bird species as they
    appear in the Clements Taxonomy 2021 revision), and some are small (all
    labels appearing in a small dataset).

*   A **mapping** provides a conversion between two namespaces. In its raw form,
    a mapping is a collection of pairs of labels, which is perfectly general
    (see below).

*   A **class list** is an ordered set of labels, from a specific namespace.
    These are useful for specifying subsets, such as classifier targets.

## Taxonomy database

The **taxonomy database** collects all of the namespace, class lists, and
mapping data into a single object for ease of access. You can instantiate the
database with `db = namespace_db.load_db()`. The database itself is cached, so
it is effectively zero-cost to load after the first time it is created in a
program.

## Data storage

The taxonomy data is stored as an SQLite database file
(`taxonomy_database.sqlite`). Under the hood, this database utilizes a
normalized relational schema to record unique strings (e.g., labels),
namespaces, class lists, and mappings. To migrate from the old JSON schema
or generate the database file, use the `convert_database.ipynb` notebook.

## Algebraic Operations and Extended Identity

Namespaces support algebraic set operations:

*   **Unions and Differences**: You can combine or subtract namespaces
    directly in Python using standard operators (e.g. `ns_c = ns_a + ns_b`
    or `ns_d = ns_a - ns_b`).
*   **Algebraic Expression Lookup**: You can fetch namespaces dynamically by
    using parseable algebraic expressions containing standard operators and
    parentheses, e.g., `db.namespaces['(ebird2021 + ebird2022) - caples']`.
*   **Extended Identity Mappings**: Mappings are constructed with automated
    default identity support. When mapping from namespace A to B, any shared
    elements `x` belonging to both A and B are mapped to themselves
    (`m(x) = x`) by default, unless overridden by the mapping pairs.
    This is helpful for defining mappings that are mostly trivial, like
    year-on-year updates of large taxonomies.

## Data consistency

When a taxonomy database is loaded, it is automatically tested for consistency.
This means, e.g., that the labels in a class list are a member of the
namespace that the class list belongs to.

## Recommendations

*   When multiple namespaces are available for a project, we recommend choosing
    a *canonical* namespace, and then converting labels to that target. Creating
    and maintaining conversion tables is a lot of work, and often error-prone.
    Given N different Namespaces, there are N² possible mappings between them.
    But with a canonical target namespace, only N mappings need to be maintained
    (in theory).

*   We also recommend leaving labels on the original data 'at rest.' This
    provides a clear record of the original label, which can then be converted
    for usage. If we later update the conversion (e.g., check in a change in a
    mapping due to a bug) or change our choice of canonical namespace (e.g.,
    update from eBird 2021 to eBird 2022), we can then be sure we haven't lost
    data.

## Details

Note that mappings are assumed to be 1:1, so each label in the source namespace
maps to exactly one label in the target namespace.

Mappings can be composed, although this should be done with care: There is no
guarantee that composed mappings give the same result as direct mappings.

## Data Sources

The taxonomy DB contains data from the
[eBird / Clements list](https://www.birds.cornell.edu/clementschecklist/download/),
the [AudioSet Ontology](https://github.com/audioset/ontology), and
[IOC World Bird List](https://www.worldbirdnames.org/new/).
The DB also contains label sets from a number of publicly available bioacoustics
datasets, with mappings to the eBird taxonomy.
