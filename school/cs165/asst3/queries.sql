-- CS 165/E-268, Assignment 3 SQL Programming Problems
-- Loren McGinnis, mcginn@fas.harvard.edu

-- 1. In which movies did Tom Cruise act?
 SELECT m.name FROM
 Person AS p, Actor AS a, Movie AS m
 WHERE p.id = a.actor_id AND m.id = a.movie_id AND p.name="Tom Cruise";

-- 2. What is the average running length of movies in each year?
 SELECT year, AVG(runtime) AS avg_runtime FROM
 Movie
 GROUP BY year;

-- 3. In which years was the average running length greater than 2 hours?
 SELECT year FROM
 	(SELECT year, AVG(runtime) AS avg_runtime FROM
 	Movie
 	GROUP BY year) AS q
 WHERE avg_runtime > 120;


-- 4. What is the average running length of a movie that wins the Best Movie Oscar?
 SELECT AVG(runtime) AS avg_runtime FROM
 Movie AS m, Oscar AS o
 WHERE o.type = 'BEST-PICTURE' AND o.movie_id = m.id;

-- 5. Which people (actors, actresses, and directors) were involved in producing "Avatar"?
 SELECT p.name FROM
 Movie AS m, Actor AS a, Person AS p
 WHERE m.id = a.movie_id AND p.id = a.actor_id AND m.name="Avatar"
 UNION
 SELECT p.name FROM
 Movie AS m, Person AS p, Director AS d
 WHERE m.id = d.movie_id AND p.id = d.director_id AND m.name="Avatar";

-- 6. Which actors and actresses acted in both "Avatar" and "Terminator Salvation"? 
 SELECT name FROM
 Person
 WHERE id IN (SELECT a.actor_id FROM
 	Movie AS m, Actor AS a
 	WHERE m.id = a.movie_id AND m.name="Avatar")
 AND id IN (SELECT a.actor_id FROM
 	Movie AS m, Actor AS a
 	WHERE m.id = a.movie_id AND m.name="Terminator Salvation");

-- 7. What percentage of directors have directed at least two movies A and B,
-- such that A and B have different genres?
 SELECT (SELECT count(*) FROM
 	(SELECT count(*) AS num_genre, director_id FROM
 		(SELECT DISTINCT director_id, genre FROM
 		Director AS d, Movie AS m
 		WHERE d.movie_id = m.id) AS q1
 	GROUP BY director_id) AS q2
 	WHERE num_genre > 1) /
 	(SELECT count(*) FROM
 	(SELECT DISTINCT director_id FROM Director) AS q3) * 100 AS percent;

-- 8. Find the names of all people who have won Oscars in two or more different
-- categories (e.g., best actor AND best director).
 SELECT p.name FROM
 Person AS p, (SELECT count(*) AS num_cat, person_id FROM
 		(SELECT DISTINCT type, person_id FROM
 		Oscar AS o) AS q
 	GROUP BY person_id) AS q2
 WHERE p.id = q2.person_id AND num_cat > 1;

-- 9. Which of the top 20 gross-earning movies did not receive the best director Oscar?
 SELECT name FROM Movie
 WHERE earnings_rank <= 20 AND id NOT IN (SELECT movie_id FROM Oscar WHERE type='BEST-DIRECTOR');

-- 10. Find all the directors with more than one movie in the top 50 grossing movies.
 SELECT p.name FROM
 Person as p, (SELECT COUNT(*) as num_movie, director_id FROM Director WHERE movie_id IN
 (SELECT id FROM Movie WHERE earnings_rank <=50) GROUP BY director_id) AS q
 WHERE num_movie > 1 and p.id = director_id;

-- 11. Who was the youngest actor to win a Best Actor award?
 SELECT name FROM (SELECT p.name, (o.year - YEAR(p.dob)) AS age FROM
 Person AS p, Oscar AS o
 WHERE p.id = o.person_id AND o.type = 'BEST-ACTOR'
 ORDER BY age) AS q
 LIMIT 1;

-- 12. Find all pairs of actors/actresses that share the same birthday
-- (month and day; the year can vary).
 SELECT p1.name AS name1, p2.name AS name2 FROM
 (SELECT * from Person where id IN (SELECT actor_id from Actor)) AS p1,
 (SELECT * from Person where id IN (SELECT actor_id from Actor)) AS p2
 WHERE (p1.id < p2.id) AND DAY(p1.dob) = DAY(p2.dob) AND MONTH(p1.dob) = MONTH(p2.dob);

-- 13. They are working their way through the decades watching all films that won
-- one of the "big 6" oscars in that decade. Produce a list of movie titles
-- for them for the 70's.
 SELECT m.name FROM
 Movie AS m
 WHERE m.year >= 1970 AND m.year < 1980 AND m.id IN
 (SELECT movie_id from Oscar AS o WHERE o.year >= 1970 AND o.year < 1980);

-- 14. Find all actors or actresses in the database who have a Bacon number of 2
CREATE TABLE BacMovie
(SELECT DISTINCT m.* FROM Movie AS m, Actor AS a, Person AS p
WHERE a.movie_id = m.id AND a.actor_id = p.id AND p.name="Kevin Bacon");
CREATE TABLE BacAct1
(SELECT DISTINCT p.* FROM Person AS p, Actor AS a, BacMovie AS m
WHERE a.movie_id = m.id AND a.actor_id = p.id);
CREATE TABLE BacMovie1
(SELECT DISTINCT m.* FROM Movie AS m, Actor AS a, BacAct1 AS p
WHERE a.movie_id = m.id AND a.actor_id = p.id);

SELECT DISTINCT p.name FROM BacMovie1 AS m, Actor AS a, Person AS p
WHERE a.movie_id = m.id AND a.actor_id = p.id AND p.id NOT IN
(SELECT id FROM BacAct1 UNION SELECT id FROM Person WHERE name="Kevin Bacon");

DROP TABLE BacMovie;
DROP TABLE BacMovie1;
DROP TABLE BacAct1;

-- Paul Erdos was a mathematician.  Similar to Bacon's Ubiquity in
-- the film industry, Erdos published more papers than any other
-- mathematician.  Hence, one's Erdos Number is the collaboration
-- distance between oneself and Erdos (working on a paper with Erdos is 1).
-- What has also arisen is the concept of a Bacon-Erdos number,
-- which is simply the sum of one's BN and EN.
-- Our very own Professor Steven "Shlomo" Gortler has a Bacon-
-- Erdos number of 5!  (Bacon-2, Erdos-3, The BE record is 3, held by
-- Daniel Kleitman)
