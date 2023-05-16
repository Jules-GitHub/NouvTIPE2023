type formula = True | False | If of int*formula*formula;;
type assignment = bool array;;
type result = Tautology | Refutation of assignment;;

let affiche_tab tab = 
  let n = Array.length tab in
  for i = 0 to n-1 do
    print_string (if tab.(i) then "1" else "0")
  done;
  print_newline()
;;

let decide1 n f =
  let v = Array.make n false in
  let rec fixe f v0 = match f with
  | True -> Tautology
  | False -> affiche_tab v0; Refutation(Array.copy v0)
  | If (i, f1, f2) -> 
    let a  = fixe f2 v0 in
    (match a with 
    | Refutation(v1) -> affiche_tab v1; a
    | _ -> v0.(i) <- true; let b = fixe f1 v0 in v0.(i) <- false; b);
  in fixe f v
;;

let rec decide2 n f = match f with
| False -> Refutation (Array.make n false)
| True -> Tautology
| If (i, l, r) ->
    (match decide2 n l with
     | Tautology -> (match decide2 n r with
                     | Tautology    -> Tautology
                     | Refutation a -> a.(i) <- false; Refutation a)
     | Refutation a -> a.(i) <- true; Refutation a)
;;

let f0 = 
  If (0,
    If (1, 
      If (2, True, False),
      False
    ),
    If (1, 
      False,
      If (2, False, True)
    )   
  )
;;

decide1 3 f0;;