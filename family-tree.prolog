/*
 * Adopted from the paper:
 * Hinton, G. E. (1986, August). Learning distributed representations of concepts. 
 * In Proceedings of the eighth annual conference of the cognitive science society (Vol. 1, p. 12).
 */

/* chr: Christopher */
/* pen: Penelope */
/* and: Andrew */
/* che: Christine */
/* mat: Margaret */
/* art: Arthur */
/* vic: Victoria */
/* jam: James */
/* jen: Jennifer */
/* chs: Charles */
/* col: Colin */
/* cha: Charlotte */
/* rob: Roberto */
/* maa: Maria */
/* pie: Pierro */
/* fra: Francesca */
/* gin: Gina */
/* emi: Emilio */
/* luc: Lucia */
/* mar: Marco */
/* ang: Angela */
/* tom: Tomaso */
/* alf: Alfonso */
/* sop: Sophia */


/* father */
father(X, Y) :- fa(X, Y).

fa(chr, art).
fa(chr, vic).
fa(and, jam).
fa(and, jen).
fa(jam, col).
fa(jam, cha).

fa(rob, emi).
fa(rob, luc).
fa(pie, mar).
fa(pie, ang).
fa(mar, alf).
fa(mar, sop).

/* mother */
mother(X, Y) :- mo(X, Y).

mo(pen, art).
mo(pen, vic).
mo(che, jam).
mo(che, jen).
mo(vic, col).
mo(vic, cha).

mo(maa, emi).
mo(maa, luc).
mo(fra, mar).
mo(fra, ang).
mo(luc, alf).
mo(luc, sop).

/* husband */
husband(X, Y) :- hu(X, Y).

hu(chr, pen).
hu(and, che).
hu(art, mat).
hu(jam, vic).
hu(chs, jen).

hu(rob, maa).
hu(pie, fra).
hu(emi, gin).
hu(mar, luc).
hu(tom, ang).

/* wife */
wi(X, Y) :- hu(Y, X).
wife(X, Y) :- wi(X, Y).

/* son */
son(X, Y) :- so(X, Y).

so(art, chr).
so(jam, and).
so(col, jam).
so(emi, rob).
so(mar, pie).
so(alf, mar).

so(art, pen).
so(jam, che).
so(col, vic).
so(emi, maa).
so(mar, fra).
so(alf, luc).


/* daughter */
daughter(X, Y) :- da(X, Y).

da(vic, chr).
da(jen, and).
da(cha, jam).
da(luc, rob).
da(ang, pie).
da(sop, mar).

da(vic, pen).
da(jen, che).
da(cha, vic).
da(luc, maa).
da(ang, fra).
da(sop, luc).

/* sex */
male(P) :- father(P, _).
male(P) :- husband(P, _).
male(P) :- son(P, _).
female(P) :- mother(P, _).
female(P) :- wife(P, _).
female(P) :- daughter(P, _).

/* parent */
parent(X, Y) :- father(X, Y).
parent(X, Y) :- mother(X, Y).

/* uncle */
uncle(X, Y) :- parent(Parent, Y), brother(X, Parent).
uncle(X, Y) :- parent(Parent, Y), sister(Aunt, Parent), husband(X, Aunt).
print_uncles :- uncle(X, Y), format('~a un ~a', [X, Y]).

/* aunt */
aunt(X, Y) :- parent(Parent, Y), sister(X, Parent).
aunt(X, Y) :- parent(Parent, Y), brother(Uncle, Parent), wife(X, Uncle).
print_aunts :- aunt(X, Y), format('~a au ~a', [X, Y]).

/* brother */
brother(X, Y) :- parent(P, Y), parent(P, X), X \= Y, male(X).
print_brothers :- brother(X, Y), format('~a br ~a', [X, Y]).

/* sister */
sister(X, Y) :- parent(P, Y), parent(P, X), X \= Y, female(X).
print_sisters :- sister(X, Y), format('~a si ~a', [X, Y]).

/* nephew */
nephew(X, Y) :- uncle(Y, X), male(X).
nephew(X, Y) :- aunt(Y, X), male(X).
print_nephews :- nephew(X, Y), format('~a ne ~a', [X, Y]).

/* neice */
niece(X, Y) :- uncle(Y, X), female(X).
niece(X, Y) :- aunt(Y, X), female(X).
print_nieces :- niece(X, Y), format('~a ni ~a', [X, Y]).