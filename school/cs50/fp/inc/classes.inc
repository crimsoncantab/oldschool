<?
//Athlete class, stores athlete attributes
    class Athlete{
        var $name;
        var $pref;
        var $pre;
        var $post;
        var $practice;

        function Athlete($athlete_, $pref_, $pre_, $post_) {
            
            $this->athlete = $athlete_;
            $this->pref = $pref_;
            $this->pre = $pre_;
            $this->post = $post_;
            $this->practice = 7;

        }

        //sets practice time to given time
        function setPractice($time) {
            $this->practice = $time;
        }

        //checks if athlete can attend a certain practice
        function isAvailable($time) {
            return ($this->pre <= $time && $this->post >= ($time + $_GET["length"]));
        }

        //checks if athlete has been added to practice or not
        function isUnassigned() {
            return ($this->practice == 7);
        }

    }

//Practice class, stores time and list of athletes
    class Practice{
        var $time;
        var $athletes;

        function Practice($time_) {
            $this->time = $time_;
            $this->athletes = array();
        }

        //fills practice with available players
        function fillPractice($Alist, $minBoat, $maxBoat, $side) {
            $tempArray = array();
            foreach ($Alist as $tempAthlete)
                if ($tempAthlete->isAvailable($this->time) && $tempAthlete->isUnassigned() && $tempAthlete->pref != 'c')
                    $tempArray[] = $tempAthlete;
            if (count($tempArray) >= $minBoat) {
                $maxBoat = min($maxBoat, count($tempArray));
                $maxBoat = $maxBoat - ($maxBoat % $minBoat);
                if ($side == "true") $this->addSides($tempArray, $maxBoat);
                for ($i = count($this->athletes), $j = 0; $i < $maxBoat && $j < count($tempArray); $i++, $j++) {
                    if ($tempArray[$j]->isUnassigned()) {
                        $tempArray[$j]->setPractice($this->time);
                        $this->athletes[] = $tempArray[$j];
                    }
                    else
                        $i--;
                }
                $this->getCox($Alist);
            }
        }

        //adds needed coxswains to practice
        function getCox($Alist) {
            $numFound = 0;
            $numCox = ceil(count($this->athletes) / 8);

            for ($i = 0; $i < count($Alist) && $numFound < $numCox; $i++)
                if ($Alist[$i]->pref == 'c' && $Alist[$i]->isAvailable($this->time) && $Alist[$i]->isUnassigned()) {
                    $Alist[$i]->setPractice($this->time);
                    $this->athletes[] = $Alist[$i];
                    $numFound++;
                }
            
            if ($numFound < $numCox)
                $this->athletes[] = new Athlete("NEEDS COX", 'c', 0, 0);
        }
        
        //adds players by sides
        function addSides($Alist, $num) {
            $this->addSide($Alist, $num / 2, 'p');
            $this->addSide($Alist, $num / 2, 's');
        }

        //adds players to a practice only for given side
        function addSide($Alist, $num, $pref) {
            $numFound = 0;

            for ($i = 0; $i < count($Alist) && $numFound < $num; $i++)
                if ($Alist[$i]->pref == $pref) {
                    $Alist[$i]->setPractice($this->time);
                    $this->athletes[] = $Alist[$i];
                    $numFound++;
                }

        }

        //returns string with practice time and athletes separated by commas
        function printPractice() {
            $string = floor($this->time);
            if ($this->time - floor($this->time) != 0)
                $string .= ":" . ($this->time - floor($this->time)) * 60;
            else
                $string .= ":00";
            $string .= " -";
            $comma = 0;
            foreach($this->athletes as $tempAthlete) {
                if ($comma) $string .= ", ";
                $string .= " " . $tempAthlete->athlete;
                if (!$comma) $comma++;
            }
            return $string;
        }
    }
?>
