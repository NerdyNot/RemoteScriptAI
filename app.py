import os
import re
import paramiko
import winrm
import yaml
import argparse
from openai import OpenAI, AzureOpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# OpenAI client setup function
def set_openai_client(openai_config):
    """
    Set up the OpenAI or Azure OpenAI client based on the configuration provided.

    Args:
        openai_config (dict): Configuration dictionary for OpenAI/Azure OpenAI.

    Returns:
        tuple: A tuple containing the client instance, client type, and model name.
    """
    client_type = openai_config['type'].strip().lower()
    api_key = openai_config['api_key'].strip()
    model = openai_config['model'].strip()
    
    if client_type == "azure":
        azure_endpoint = openai_config['azure_endpoint'].strip()
        azure_apiversion = openai_config['azure_apiversion'].strip()
        os.environ['AZURE_OPENAI_API_KEY'] = api_key
        return AzureOpenAI(
            api_version=azure_apiversion,
            azure_endpoint=azure_endpoint
        ), client_type, model
    else:
        os.environ['OPENAI_API_KEY'] = api_key
        return OpenAI(api_key=api_key), client_type, model

console = Console()

# Function to run command on Linux server using paramiko
def run_remote_command_linux(host, port, username, password, command):
    """
    Run a command on a remote Linux server using Paramiko SSH client.

    Args:
        host (str): The hostname or IP address of the server.
        port (int): The port number for SSH connection.
        username (str): The username for SSH authentication.
        password (str): The password for SSH authentication.
        command (str): The command to execute on the remote server.

    Returns:
        str: The output from the command execution.
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username=username, password=password)
    stdin, stdout, stderr = ssh.exec_command(command)
    output = stdout.read().decode('utf-8') + stderr.read().decode('utf-8')
    ssh.close()
    return output

# Function to run command on Windows server using pywinrm
def run_remote_command_windows(host, port, username, password, command):
    """
    Run a command on a remote Windows server using WinRM.

    Args:
        host (str): The hostname or IP address of the server.
        username (str): The username for WinRM authentication.
        password (str): The password for WinRM authentication.
        command (str): The PowerShell command to execute on the remote server.

    Returns:
        str: The output from the command execution.
    """
    endpoint = f'http://{host}:{port}/wsman'
    session = winrm.Session(endpoint, auth=(username, password), transport='ntlm')
    result = session.run_ps(command)
    return result.std_out.decode('utf-8') + result.std_err.decode('utf-8')

# Function to generate script based on user input
def generate_script(user_input, client_type, model, os_info, shell_version, os_type):
    """
    Generate a script based on user input and system information using OpenAI or Azure OpenAI.

    Args:
        user_input (str): The user's input query or command.
        client_type (str): The type of OpenAI client (azure or openai).
        model (str): The model name to use for the OpenAI request.
        os_info (str): Information about the operating system.
        shell_version (str): The shell version on the operating system.
        os_type (str): The type of operating system (Linux or Windows).

    Returns:
        str: The generated script.
    """
    system_prompt = f"""
    # Instruction
     - You are an assistant for a {os_type} operating system with the following details
     - You must only provide the Script, without any addtional explanation or text. like description or ```bash or ```powershell or ```sh.
     - Your responses should be informative, visually appealing, logical and actionable.
     - Your responses showld be very simply and completion.

    # Script Creation Rules
     - OS Information: {os_info}
     - Shell Version: {shell_version}
     - Based on the user's input, generate a script to accomplish the task.
     - To distinguish between each server, print the hostnames on environment variables.
    """

    response = None
    if client_type == "azure":
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_input}
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_input}
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

    full_response = response.choices[0].message.content.strip()
    # Extract script part
    script_match = re.search(r'```(powershell|bash|zsh|sh)\s(.*?)\s```', full_response, re.DOTALL)
    if script_match:
        script = script_match.group(2).strip()
    else:
        script = full_response  # Use the entire response if the script is not detected

    return script

# Function to explain execution result in natural language
def explain_execution_result(user_input, results, client_type, model):
    """
    Explain the execution results of a script in natural language using OpenAI or Azure OpenAI.

    Args:
        user_input (str): The user's input query or command.
        results (list): The list of results from script execution on multiple servers.
        client_type (str): The type of OpenAI client (azure or openai).
        model (str): The model name to use for the OpenAI request.

    Returns:
        str: The explanation of the results.
    """
    combined_results = "\n".join(results)
    prompt = f"""
    You need to provide a detailed explanation of the results of executing a script.
    The user should not know that the explanation is based on the script's results.
    ```
    User Query: {user_input}
    Script Execution Results: {combined_results}
    ```
    Refer to the script execution results to respond simply to the user's query.
    If the query is simply to run a specific program, respond that the program has been executed.
    Always respond in the user's language.
    """

    response = None
    if client_type == "azure":
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt.strip()}
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt.strip()}
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

    explanation = response.choices[0].message.content.strip()
    return explanation

def main():
    """
    Main function to execute the remote AI script executor application.
    Parses command-line arguments, loads configuration, sets up OpenAI client,
    and processes user input for executing commands on remote servers.
    """
    parser = argparse.ArgumentParser(description="Remote AI Script Executor")
    parser.add_argument("-c", "--config", required=True, help="Path to the configuration YAML file")
    parser.add_argument("-y", "--yes", action="store_true", help="Run without confirmation")
    parser.add_argument("-g", "--group", help="Specify the server group")
    parser.add_argument("-s", "--server", help="Specify the server alias for single server execution")
    parser.add_argument("-q", "--query", help="Specify the query or command")
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    openai_config = config['openai']
    server_groups = config['servers']

    # Set up the OpenAI client
    global client, client_type, model
    client, client_type, model = set_openai_client(openai_config)

    # Validate the selected group if specified
    if args.group and args.group not in server_groups:
        console.print("[bold red]Invalid group selected.[/bold red]")
        return

    # Handle single server selection or group selection
    if args.group == 'single' or args.server:
        all_servers = [server for group in server_groups.values() for server in group]
        if args.server:
            selected_server = next((server for server in all_servers if server['alias'] == args.server), None)
            if not selected_server:
                console.print("[bold red]Invalid server selected.[/bold red]")
                return
            servers = [selected_server]
        else:
            print("Available servers:")
            for server in all_servers:
                console.print(f"- {server['alias']}")
            
            selected_server_alias = input("Please select a server by alias: ").strip()
            selected_server = next((server for server in all_servers if server['alias'] == selected_server_alias), None)
            
            if not selected_server:
                console.print("[bold red]Invalid server selected.[/bold red]")
                return
            
            servers = [selected_server]
    else:
        if args.group:
            servers = server_groups[args.group]
        else:
            print("Available groups:")
            for group in server_groups.keys():
                console.print(f"- {group}")
            selected_group = input("Please select a group: ").strip()

            if selected_group not in server_groups:
                console.print("[bold red]Invalid group selected.[/bold red]")
                return

            servers = server_groups[selected_group]

    query = args.query
    if not query:
        print("Please enter your command or question below (type 'exit' or 'quit' to exit):")

    while True:
        if not query:
            user_input = input(">>> ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting the program. Goodbye!")
                break
        else:
            user_input = query

        if user_input:
            linux_servers = []
            windows_servers = []

            for server in servers:
                host = server['host']
                port = server.get('port', 22 if server['os_type'] == 'Linux' else 5985)  # Default ports based on OS type
                username = server['username']
                password = server['password']
                os_type = server['os_type']

                if os_type == 'Linux':
                    os_info = run_remote_command_linux(host, port, username, password, "uname -a")
                    shell_version = run_remote_command_linux(host, port, username, password, "echo $SHELL --version")
                    linux_servers.append((host, port, username, password, os_info, shell_version))
                elif os_type == 'Windows':
                    os_info = run_remote_command_windows(host, port, username, password, "systeminfo")
                    shell_version = run_remote_command_windows(host, port, username, password, "$PSVersionTable.PSVersion")
                    windows_servers.append((host, port, username, password, os_info, shell_version))
                else:
                    console.print(f"[bold red]Unsupported OS type: {os_type}[/bold red]")

            while True:
                results = []

                # Generate and execute script for Linux servers
                if linux_servers:
                    os_info = linux_servers[0][4]
                    shell_version = linux_servers[0][5]
                    linux_script = generate_script(user_input, client_type, model, os_info, shell_version, "Linux")
                    console.print(Panel(f"[bold cyan]Generated Script for Linux servers:[/bold cyan]\n\n[bold]{linux_script}[/bold]"))

                # Generate and execute script for Windows servers
                if windows_servers:
                    os_info = windows_servers[0][4]
                    shell_version = windows_servers[0][5]
                    windows_script = generate_script(user_input, client_type, model, os_info, shell_version, "Windows")
                    console.print(Panel(f"[bold cyan]Generated Script for Windows servers:[/bold cyan]\n\n[bold]{windows_script}[/bold]"))

                # Confirm and execute the command on all servers in the selected group
                if args.yes:
                    confirm = 'yes'
                else:
                    confirm = input("Do you want to execute this command on all servers in the selected group? (yes/y/no/n/regeneration/re): ").strip().lower()

                if confirm in ['yes', 'y']:
                    for host, port, username, password, os_info, shell_version in linux_servers + windows_servers:
                        if os_info.startswith('Linux'):
                            output = run_remote_command_linux(host, port, username, password, linux_script)
                        else:
                            output = run_remote_command_windows(host, port, username, password, windows_script)

                        results.append(f"Output from {host}:\n{output}")

                    if results:
                        explanation = explain_execution_result(user_input, results, client_type, model)

                        # Displaying combined results
                        combined_results = "\n\n".join(results)
                        console.print(Panel(f"[bold green]Combined Command Execution Output:[/bold green]\n\n{combined_results}"))

                        # Displaying explanation
                        md = Markdown(explanation)
                        console.print(Panel(md, title="[bold yellow]Explanation[/bold yellow]"))
                    break

                elif confirm in ['regeneration', 're']:
                    console.print("[bold yellow]Regenerating the script...[/bold yellow]")
                else:
                    console.print(f"[bold red]Command execution canceled.[/bold red]")
                    break
        else:
            console.print("[bold red]Please enter a query.[/bold red]")
            query = None

if __name__ == "__main__":
    main()