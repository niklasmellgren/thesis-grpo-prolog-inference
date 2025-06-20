%%writefile prolog_helpers.pl
:- module(prolog_helpers, [analyze_code/3]).
:- use_module(library(readutil)).

%% analyze_code(+File, -PredicateCount, -ConstraintCount)
%  Reads the Prolog source in File, counting:
%    - Predicates (i.e., top-level clauses) except `solve/1`
%    - Curly-brace constraints (anywhere in the term)
%  Then prints these counts as:
%    PREDICATE_COUNT: <num>
%    CONSTRAINT_COUNT: <num>
analyze_code(File, PredicateCount, ConstraintCount) :-
    open(File, read, Stream),
    read_terms(Stream, Terms),
    close(Stream),
    count_predicates(Terms, PredicateCount),
    count_constraints(Terms, ConstraintCount),
    format('PREDICATE_COUNT: ~w~n', [PredicateCount]),
    format('CONSTRAINT_COUNT: ~w~n', [ConstraintCount]).

%% read_terms(+Stream, -Terms)
%  Reads terms until end_of_file, returning them in a list.
read_terms(Stream, Terms) :-
    read_term(Stream, Term, [variable_names(_)]),
    ( Term == end_of_file ->
         Terms = []
    ; read_terms(Stream, Rest),
      Terms = [Term|Rest]
    ).

%% count_predicates(+Terms, -Count)
%  Among all top-level clauses, exclude `solve/1`.
count_predicates(Terms, Count) :-
    include(valid_predicate, Terms, ValidPreds),
    length(ValidPreds, Count).

valid_predicate(Term) :-
    % Skip directives (:- operator) first
    \+ Term = (:- _),
    get_head(Term, Head),
    nonvar(Head),
    Head =.. [Functor|_],
    Functor \= solve.  % Exclude solve/1

%% get_head(+Term, -Head)
%  If it's (Head :- Body), unify Head. Otherwise, it's a fact, so unify Term.
%  Skip directives
get_head((Head :- _), Head) :- !.
get_head(Head, Head) :-
    % Additional check to skip directive heads
    \+ Head = (:- _).

%% count_constraints(+Terms, -Count)
%  Count all curly-brace constraints in all terms.
count_constraints(Terms, Count) :-
    aggregate_all(count, (member(Term, Terms), has_constraint(Term)), Count).

has_constraint(Term) :-
    contains_constraint(Term).

%% contains_constraint(+Term)
%  Recursively checks sub-terms for { ... } patterns.
contains_constraint(Term) :-
    nonvar(Term),
    (  Term = {_}                % direct curly-brace
    ;  Term =.. [_|Args],
       member(Arg, Args),
       contains_constraint(Arg)
    ).
