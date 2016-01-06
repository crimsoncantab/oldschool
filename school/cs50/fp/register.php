<? 
    // require common code
    require_once("inc/common.inc"); 
?>

<!DOCTYPE html 
     PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
     "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">

<html xmlns="http://www.w3.org/1999/xhtml">

  <head>
    <title>Register</title>
  </head>

  <body>
    <div align="center">
      <form action="register2.php" method="post">
        <table border="0">
          <tr>
            <td class="field">Name:</td>
            <td><input name="name" type="text" /></td>
          </tr>
          <tr>
            <td class="field">Username:</td>
            <td><input name="username" type="text" /></td>
          </tr>
          <tr>
            <td class="field">Password:</td>
            <td><input name="password" type="password" /></td>
          </tr>
          <tr>
            <td class="field">Confirm Password:</td>
            <td><input name="password2" type="password" /></td>
          </tr>
        </table>
        <input type="submit" value="Register" />
      </form>
    </div>

  </body>

</html>
